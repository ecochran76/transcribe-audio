# Managed transcription design interview

Status: INTERVIEWING. Accepted decisions below govern the design; unanswered
recommendations remain proposals. This interview does not authorize implementation,
model trials or runtime changes.

## Established operator constraints

- Voice recordings continue through AssemblyAI with diarization.
- Screen-recording regeneration should use local hardware.
- Background processing must not monopolize the GPU outside resource management.
- Screen processing remains paused and direct GPU fallback remains disabled.
- This session discusses LocalAI qualification and Transcribe Audio integration;
  no qualification/model run or route change is authorized by the interview.

## Source anchors

- VISION.md: durable diarized utterances are outcome1; running a model alone does
  not establish the complete conversation-intelligence outcome.
- CONTEXT.md is the existing Transcribe Audio domain vocabulary authority.
- docs/dev/plans/0093-2026-11-09-managed-localai-pilot-consumer.md: existing managed
  client source acceptance; native/runtime acceptance belongs to LocalAI.
- managed_transcription.py: durable request identity, observation after uncertain
  submission, whole-file bounds and application-owned output guard.
- LocalAI qualification/GLOSSARY.md and current qualification authorities govern
  manager terminology and acceptance; reconcile before proposing new glossary terms.

## Round1 accepted decisions

The operator answered yes to all three proposals:

1. First prove managed local transcription AND diarization on a small,
   representative screen-recording set before broad backlog regeneration.
2. Start with Parakeet v3 plus Pyannote Community-1 as qualification candidates;
   either can be replaced if quality, feature or resource evidence fails.
3. Historical screen processing yields to higher-priority demand, even if that
   substantially extends backlog completion time. Yield delay and restart
   behavior still require agreement.

The operator explicitly places heavy historical retranscription and diarization
below recent voice transcription, search/retrieval workers, live transcription
and live text-to-speech. This establishes their relation to historical work,
not a priority order among those four protected workloads.

## Round2 accepted decisions

The operator answered yes to Q4-Q6:

4. Effective workload precedence: live transcription/live TTS, then interactive
   search/retrieval, then recent voice-recording transcription, then historical
   reprocessing. Ranking does not itself supply admission or preemption guarantees.
5. Historical work must never age above the protected classes. It may remain
   starved for hours under sustained protected demand; waiting reasons must be
   visible and work resumes automatically when capacity becomes available.
6. Newly discovered or newly synced old recordings remain historical; discovery
   time alone does not confer recent-work priority.

This is intended policy. Existing SharedService background promotion is a known
implementation mismatch, not a change already applied. AssemblyAI voice watching
and the paused historical screen route remain unchanged during the interview.

## Round3 accepted decisions

The operator answered yes to Q7-Q10:

7. Initial mixed-load responsiveness goals are500ms embedding response,1second
   to first TTS audio, and2seconds for live transcription. Model-profile
   performance and manager-added delay must be measured separately. These are
   desired qualification goals, not current implemented guarantees. Percentiles,
   warm/cold cohorts and precise timing boundaries still need to be frozen.
8. Historical processing may be interrupted and its unfinished bounded unit
   discarded, while completed checkpoints survive. Resume from checkpoints;
   reconcile uncertain outcomes before retrying. Cancellation alone is not exit
   proof or permission to duplicate an in-flight request.
9. Anonymous speaker labels must remain consistent across a recording, including
   chunks and restarts; independently numbered chunk labels are not identity proof.
10. Initially at most one historical GPU stage may execute at a time. Coexistence
    with protected work requires demonstrated headroom and responsiveness; more
    historical concurrency requires new measurements.

## Round4 decisions and Q11 refinement

11. Unconditional always-warm residency was NOT accepted. Protection needs a
    precise meaning. The operator explicitly wants quiet periods, such as overnight
    when humans are inactive, available for long historical workloads. This does
    not reverse the precedence of an actually active live/interactive workload.
    Quiet-mode entry, reclaimable residency and foreground-return behavior remain
    unresolved; do not interpret static protected=true as the desired final policy.
12. Accepted resumable work units for long recordings, followed by recording-wide
    speaker reconciliation before publishing the final transcript. Preserve durable
    raw/intermediate evidence and explicitly unresolved attributions.
13. Accepted a small representative pilot with reviewed audio excerpts assessing
    transcription accuracy, speaker attribution and timing. Existing transcripts
    and speaker-count agreement alone are not truth labels.
14. Accepted the initial response goals as p95 warm-request goals under mixed load.
    Cold starts and worst-case delays are reported separately. Queuing/reloads
    caused by historical work remain visible, not removed to improve the score.

See ADR0005 for the accepted resumable-recording boundary. This ADR records design,
not implementation or deployment authority.

## Round5 frontier (unanswered)

- Definition of protection: active sessions versus idle warm-readiness preferences.
- Quiet-mode entry: observed service activity/session state versus fixed schedule;
  exact inactivity interval and manual override.
- Historical access to idle model residency during quiet mode.
- Foreground-return behavior and a separately bounded cold-start experience.

After these settle, remaining branches are concrete pilot/quality thresholds,
error and retry limits, rollout/stop criteria and final shared-understanding review.
No runtime/model effects have occurred in this interview.

## Verified LocalAI contract and qualification readback

The LocalAI source readback is main f4bcf1389, with an existing untracked0161
foreground diagnosis retained untouched. Existing shared-service ownership is
already specified in qualification/docs/specs/0003-shared-consumer-integration.md:
application owns work units/checkpoints/canonical outputs; shared service owns
admission and GPU residency/lifecycle. It is not an unanswered interview decision.

Current0148 selected management evidence is1305/1305 requests with22/22
resource/correctness/lifecycle checks over30minutes; bounded-canary advice retains
latency exceptions.0160 useful canonical/foreground contention gates remain open.
Existing spec0004 selects faster-whisper without diarization, so it does not qualify
the newly requested Parakeet/Pyannote pipeline. Parakeet has bounded synthetic
managed evidence; no Community-1 managed qualification was found. Dated pilot
active claims are superseded by the operator-authorized pause in this conversation.
These source records do not constitute a fresh runtime acceptance or permission
to start model trials. See LocalAI qualification/docs/tickets/0148*,0160* and
qualification/docs/specs/0004-transcribe-managed-pilot.md.

## Round2 factual prerequisite

Read-only LocalAI source inspection found SharedService workload defaults live-stt0,
foreground1 and background2. Its dispatcher promotes a waiting background job to
priority-1 after three interactive starts (qualification/inference_manager/
shared_service.py:54-68,82-88,190-207). This is distinct from the standalone
scheduler aging rule. Q5 now rejects this promotion for historical reprocessing; no service behavior
was changed during this interview.
Existing yield is at work-unit boundaries, not GPU-kernel preemption, and does not
by itself release residency. The old pilot has whole-file units, not an accepted
mid-file checkpoint or cross-chunk speaker-continuity contract.

## Round4 factual prerequisites

Existing protected residency excludes a profile from automatic idle eviction;
it is not itself a timed warm lease. Shared authority must retain ownership of
Speech session/warm-lease lifecycle. Capacity admission already rejects missing,
stale (>5second) or failed telemetry and requires reservation/free-capacity fit;
those safety floors are retained. The16GiB host's12GiB planning target and roughly
4GiB headroom where feasible are not an unconditional static partition.

Existing pilot observation concepts (matched direct baseline, canonical outputs,
foreground cohort, resource/lifecycle evidence) can be reused, but old acceptance
cannot be transferred to Parakeet/Pyannote with chunking and interruption. It needs
its own declared cohort and diarization-quality rubric. Existing observer Boolean
quality fields alone do not establish word, turn or speaker continuity quality.

## Quiet-time factual readback

The current shared service has no common quiet-hours/activity detector, idle
expiry or timed warm lease. The pilot pins foreground Nomic with static
protected=True. Active-session protection and a warm-cache preference must not
be conflated. Existing documentation wants both warm interactive usefulness
and reclaiming idle models; it does not require unconditional always-warm operation.

A retained0161 diagnostic recorded25.43seconds embedding startup; this is one
observation, not a cold p95 or promised reload time. Warm Nemotron evidence and
one-shot complete-WAV Qwen evidence do not establish a common cold-return budget.
Quiet eligibility and the first interactive request after eviction therefore
need separate decisions. No activity collection, idle policy or new memory
reservation has been implemented during this discussion.
