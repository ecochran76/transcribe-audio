# Plan 0090 | Audio-only AssemblyAI uploads

State: CLOSED
Scope: Prepare audio-only uploads for video recordings in the AssemblyAI backend.
Owner: Primary coordinator
Integration target: main
Source base: 88b7051599fe458328aaee734859bc1f1393c014
Tracker: Repo-native numbered plans, ROADMAP.md and RUNBOOK.md per repository policy.
Review status: Operator approved test seam and one-ticket breakdown on 2026-11-01.

## Current State

The AssemblyAI backend streams each original recording to the upload endpoint,
including video tracks. Watched video folders use that same backend. Local
faster-whisper is the existing watcher fallback. Implementation is complete on the integration branch; local validation passes.

## Problem Statement

Screen recordings carry video data that speech transcription does not need.
Uploading that data increases transfer volume and delays the transcription
pipeline. Observed allocation failures have not been causally attributed to
video uploads; this change does not promise to resolve those failures.

## Solution

Extract temporary lossless audio from video recordings before AssemblyAI upload.
Keep the original recording as the authority for calendar association, naming,
exports and artifact provenance. Clean up temporary audio on success or failure.
If extraction fails, fail the AssemblyAI attempt clearly so the existing watcher
can use its configured backend fallback. Never silently upload the original
video after extraction failure. Audio-only input retains its existing behavior.

## User Stories

1. As an operator, I want only audio sent for video transcription, so that unnecessary video data does not leave the machine.
2. As an operator, I want smaller uploads, so that screen recordings consume less upstream bandwidth.
3. As an operator, I want lossless audio extraction, so that preparation introduces no additional lossy encoding.
4. As an operator, I want original recordings preserved, so that I can replay or reprocess them.
5. As an operator, I want calendar association to use original recording metadata, so that temporary-file timestamps do not change the selected event.
6. As an operator, I want existing calendar renaming behavior preserved, so that outputs stay associated with their recording.
7. As an operator, I want transcript artifacts to identify original recordings, so that provenance does not point to deleted temporary files.
8. As an operator, I want temporary audio removed after success, so that the backlog does not accumulate private working files.
9. As an operator, I want temporary audio removed after failure, so that failed uploads do not leave unnecessary private copies.
10. As an operator, I want extraction failures to be visible, so that I can distinguish preparation errors from provider errors.
11. As an operator, I want the watcher to retain its existing backend fallback, so that one provider preparation failure does not disable local transcription.
12. As an operator, I want audio-only recordings to keep working, so that this video improvement does not change the established mobile-recording workflow.
13. As an operator, I want preparation to write audio to disk with bounded subprocess execution, so that entire decoded recordings are not buffered in Python memory.
14. As an operator, I want local synthetic-media validation, so that correctness can be checked without uploading private meetings or spending provider quota.

## Implementation Decisions

- Keep provider-specific preparation inside the AssemblyAI backend, behind its existing processing interface.
- Use ffmpeg to select the first audio stream, omit video, and emit a temporary FLAC derivative without additional lossy encoding. Preserve sample rate and channel count.
- Apply preparation to supported video container extensions; preserve direct uploading for established audio-only extensions.
- Use a private temporary directory with automatic cleanup and a bounded ffmpeg subprocess. Report missing tools, absent audio streams, failed extraction and timeout as transcription errors.
- Stream the prepared file rather than reading the entire derivative into memory.
- Pass the original recording to the unchanged output/calendar pipeline.
- Keep existing watcher fallback policy; no new provider, store schema, CLI option, or private-data routing.

## Testing Decisions

Approved primary seam: the existing AssemblyAI recording-processing interface,
with a local fake HTTP/provider boundary and real short synthetic media.
Observe uploaded bytes and generated outputs; avoid tests of internal call order.
This catches video leakage, original-recording provenance regressions, cleanup,
visible failures and audio-only regressions. It does not prove live provider
acceptance, historical calendar correctness or memory-incident remediation.

Use the existing pytest harness and transcript artifact/export tests as prior art.
Run red/green behavior slices, then affected watcher/output/calendar tests. A
short CLI smoke must emit DOCX, TXT and transcript JSON using synthetic media and
a local fake provider; inspect the uploaded media independently with ffprobe.
Hosted GitHub Actions are excluded by operator CI cost policy.

## Non-Goals

No memory incident diagnosis, backlog reset, automatic historical retranscription,
new fallback policy, identity work, video editing, service restart, live provider
smoke, release, remote publication or merge authorization.

## Out of Scope

Multiple-audio-stream mixing or selection UI, new media formats, compression
quality tuning, retention-policy changes, and changes to local transcription.
Original recording preservation permits existing explicit calendar renaming;
it forbids deleting or replacing the original with the audio derivative.

## Vision Outcomes

Advances durable transcript yield and recording/calendar provenance as
prerequisites for contextual readouts. Audio-only video-upload preparation moves
from maturity 0 (absent) to 1 (built), measured by local synthetic-media acceptance.
Operational maturity 3 requires later observed automatic processing evidence;
this source packet does not claim it or complete the conversation knowledge loop.

## Acceptance Criteria

- Video inputs upload decodable audio-only bytes; the selected source audio remains lossless through preparation.
- Original recording content remains intact and original metadata drives calendar/export handling.
- DOCX, TXT and sidecar exports retain original recording association.
- Audio-only inputs retain current upload behavior.
- Temporary derivatives are removed after success, upload failure and extraction failure.
- Missing ffmpeg, no audio stream and extraction timeout fail clearly without uploading original video.
- Existing watcher fallback remains usable through the existing nonzero backend failure contract.
- Focused automated tests and a synthetic-media CLI smoke pass locally.

## Proposed Ticket Breakdown

1. **Audio-only uploads with original recording provenance**
   - **Blocked by:** None; can start immediately after this review.
   - **Delivers:** Complete preparation/upload/export behavior, cleanup and failure handling, unchanged audio-only input and watcher fallback, documentation, regression tests and synthetic CLI validation.
   - **Granularity:** One vertical ticket because all behavior shares one provider boundary; no prefactoring or parallel writers are needed.

Publish one separate numbered ticket after operator review of granularity,
blocking edges and the test seam. Then create one integration branch from the
recorded base, execute sequentially and keep a resumable ticket ledger here.

## Validation

Pending. Source-only validation uses no real meeting media or live API calls.
Record exact commands, outcomes, source commit and working patch identity.
Review Standards and Spec separately against the frozen base, with at most one
broad review and one bounded remediation pass. Local results satisfy CI; fake
provider validation differs materially from hosted/live provider environments.

## Further Notes

The integration branch is the delivery boundary. Tracker resolution, main
integration and installed-service activation are separate states. Existing
watcher authorization remains in force; this spec does not reset its state.

Memory disposition: forbidden; developer instructions require explicit user
requests before updating agent memory, and this task requests engineering work.

## Execution Ledger

Integration branch: `feat/assembly-audio-only`
Base: `88b7051599fe458328aaee734859bc1f1393c014`
Ticket: Plan0091; blockers: none; state: validated.
Worker: primary; sequential execution, no worker branches.
Owned paths: AssemblyAI backend, focused upload tests, README, CONTEXT,
Plan0090/0091, ROADMAP and RUNBOOK.
Bounds: one broad review and one remediation pass; local focused tests and
selected regression tests; no live provider traffic or service changes.

## Local Validation Receipt | 2026-11-01

Tier: focused plus selected presubmit regression; primary execution, one process,
no shards, no live/soak or comprehensive-suite claim. No hosted GitHub Actions.
Environment: Linux/WSL, existing project Python 3.12 environment, ffmpeg 8.1,
ffprobe, real synthetic media and fake provider responses. Tests require ffmpeg
on PATH; prefix commands with `PATH=/home/linuxbrew/.linuxbrew/bin:$PATH` on this host.
No production media, calendar credentials, GPU model or provider quota used.

- Red: `python -m pytest -q tests/test_assembly_audio_upload.py` initially failed
  at fixture startup because ffmpeg was absent from this worktree shell PATH.
  After explicit PATH correction, it failed at the intended boundary:
  uploaded streams were `[video, audio]`, expected `[audio]`.
- Green: the same focused command passed after implementing extraction.
- Selected regression: `python -m pytest -q tests/test_assembly_audio_upload.py
  tests/test_transcript_artifacts.py tests/test_readouts.py`: **62 passed in 2.29s**.
  Includes eight new cases, watcher retry/readiness, artifact/export and calendar
  regression coverage. No failed-test retry was hidden: one new calendar test
  initially expected a provider event ID omitted by the existing export schema;
  its assertion was corrected to the exported summary. This was a test-authoring
  error, not a flaky product failure.
- `python -m py_compile assembly_transcribe.py tests/test_assembly_audio_upload.py`: pass.
- `git diff --check`: pass.
- Manual synthetic MP4 CLI: `python ~/.local/state/transcribe-audio/audio-upload-smoke.py`:
  pass with fake upload/transcript/poll responses; DOCX text, TXT and sidecar checked.
  Source **542699 bytes**, uploaded FLAC **111664 bytes**, independently inspected
  as one audio stream with no video. Original SHA-256 unchanged.
  Smoke receipt/log and synthetic media are user-scoped under
  `~/.local/state/transcribe-audio/audio-upload-smoke/`; the replayable script is
  its sibling `audio-upload-smoke.py`. This is deliberately local evidence,
  not a server or provider artifact.
- Real decoded PCM equality checked for the lossless synthetic source; original
  recording mtime, source/working paths and calendar rename are tested.
- Failure cases cover absent audio, missing executable, extraction timeout,
  upload failure after opening the derivative, and missing provider upload URL.
  Cleanup and nonzero CLI failure preserve the existing watcher fallback contract.

## Standards Review

Primary reviewed the complete source/test/doc patch against recorded base
88b7051599fe458328aaee734859bc1f1393c014, AGENTS.md, runtime-state,
architecture, testing, Git and validation policy. Provider-specific preparation
stays inside its backend; no schema, state routing or unrelated work changes.
No accepted blocking findings. One broad primary review; no delegation and no
remediation pass needed.

## Spec Review

Primary independently checked Plan0090 acceptance against the same patch and
local evidence: audio-only transport, lossless preparation, original provenance,
calendar/export behavior, failure cleanup, direct audio-only upload and fallback
contract are covered. No accepted blocking findings. Live provider acceptance,
operational backlog yield and the memory incident remain explicitly unproved.

## Delivery State

Ticket0091: implemented and integrated in the feature checkout; validation passed.
Source result SHA: `1dcea2797704827a5c022da972158a1ad1315f79`.
Ticket dependency graph: one node, no missing blocker IDs or cycles.
Integration: source commit is on `feat/assembly-audio-only`; source hash matches
manual smoke receipt. Tracker: local Ticket0091 resolved CLOSED on source acceptance.
Final tip is the documentation closeout commit containing this ledger; resolve
`git rev-parse feat/assembly-audio-only`. No worker handle/branch/result applies;
the primary owns the complete write scope. Validation result remains valid
because this follow-up changes documentation only.
No main merge, remote publication, live provider smoke or installed-service change.
