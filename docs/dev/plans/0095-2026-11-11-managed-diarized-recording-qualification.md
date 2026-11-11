# Managed diarized recording qualification and integration

State: OPEN
Owner: primary integration coordinator
Scope: qualify a bounded resumable local transcription-and-diarization pilot
Execution: source implementation validated; native acceptance blocked on bounded owned release and remaining native cohorts
Source baseline: Transcribe Audio main, 61471db; LocalAI codex/0157-transcribe-pilot, f4bcf138967f570c7bd0c9646800be28b338aa62
Decision provenance: [interview](../notes/0062-2026-11-10-managed-transcription-design-interview.md), Q1-Q18
Architecture: [ADR0005](../../adr/0005-resume-recordings-across-managed-inference-units.md)

## Objective and vision outcome

Produce durable, useful diarized transcripts from historical screen recordings on
local hardware through the existing Shared GPU service, preserving responsiveness
for current human-facing work. Advance VISION outcome 1 (durable transcript and
diarized utterances), evidence integrity and autonomy: the combined managed route
is currently level 1 (built and tested), while component native ASR/diarization
have bounded partial shadow evidence; full managed native acceptance remains open. Target level 2 for this bounded
managed pilot, with level 3 requiring subsequent measured automatic operation.

Measure canonical output yield, transcription/speaker/timing quality, recording-wide
speaker continuity, protected-work response time, manager-added cost, safe resource
release and recovery without duplicate publication. These outputs feed the existing
conversation-intelligence loop; this plan does not claim to realize that whole loop.

## Scope

- Reuse SharedService/Manager and existing authenticated consumer contracts.
- Qualify Parakeet TDT 0.6B v3 plus Pyannote Community-1 as starting candidates,
  replacing either only through an explicitly revised profile/evidence comparison.
- Add managed ownership for every GPU stage, including diarization and any GPU
  speaker reconciliation; retain application ownership of recording progress and
  canonical artifacts.
- Implement agreed priority, automatic protection, quiet-period reclamation and
  explicit operator overrides through the shared authority.
- Support bounded resumable long-recording units and recording-wide speaker labels.
- Finish with a small representative managed pilot and a truthful acceptance report.

## Non-Goals

- Starting the full historical backlog or silently enlarging the pilot cohort.
- Replacing the active AssemblyAI voice-recording route or uploading the historical
  screen corpus to a cloud provider.
- Installing Slurm, adding another scheduler to Transcribe Audio, or treating a file
  lock as shared GPU admission.
- Named speaker identity, cross-recording person matching, full readouts or knowledge
  projection qualification.
- Claiming whole-LocalAI acceptance from this consumer pilot, reusing exhausted
  native trial allowances, pausing unrelated services, raising resource ceilings,
  resetting the GPU, or paid hosted CI.

## Initial source state and authority (preserved)

- Voice/Sound Recordings routes use AssemblyAI with diarization and an explicit
  private environment credential selector. The requested three recent voice
  retranscriptions completed; historical screen processing is disabled.
- Direct Parakeet/faster-whisper fallback is disabled in active watcher configs.
- Plan 0093's managed client supports bounded whole-file ASR and durable request/output
  guards. Its accepted native profile used faster-whisper without diarization; that
  evidence cannot qualify this new combined route.
- Parakeet's checkpoint matches a LocalAI declared asset and has bounded synthetic
  managed evidence. Pyannote lacks a qualified shared native adapter/profile.
- Current SharedService promotes background work after three interactive starts;
  that conflicts with the newly accepted historical-work precedence. Current profile
  protection is static, not the agreed activity/quiet-period policy.
- LocalAI 0148 supports a selected bounded-canary recommendation with retained latency
  exceptions;0160's useful-output/foreground-contention gates remain open. Preserve
  the existing untracked0161 diagnostic and all negative evidence. Earlier interview
  text calling this checkout main was corrected to codex/0157-transcribe-pilot.
- LocalAI qualification specs0001/0003, its GLOSSARY, existing plan0014 and tickets
  own service semantics and portfolio acceptance. This document owns this consumer
  milestone; it does not override those authorities or rewrite their historical gates.

## Current State

- Isolated integration branch `codex/managed-diarized-0095` contains the historical
  precedence/quiet policy, owned stage workers, resumable processor, guarded CLI,
  recording-wide speaker reconciliation and qualification/evidence tooling.
- LocalAI source regression:352tests OK with49optional/live skips; joined app70
  tests pass. These are source checks, not native acceptance. Later focused
  controller/provider changes have separately retained red/green evidence.
- Native CPU streaming Phonon and public AMI reference prerequisites are complete.
  Optional CPU BGE strategy was measured and rejected for the unchanged500ms
  target; GPU embedding candidate remains unmeasured after one interrupted load.
- Real Nemotron3GiB canary produced an incremental transcript, but original5s
  capacity acquisition contract failed under load before a20warm paired cohort.
  Full paired ASR and combined useful-output acceptance remain open.
- Diagnostic sampler progress has CPU/provider proof that fresh diagnostic data
  cannot refresh stale GPU counters. An isolated model-free source qualification
  qualified cached-factory0180 source achieved240unique valid idle samples. Its
  subsequent DIA run maintained fresh acquisition; no freshness/headroom bypass.
- Production voice/main watchers were freshly verified active, selected jobs
  Assembly-only and screen processing disabled. No existing service/model was
  paused, reconfigured or displaced. Historical negative captures are retained.
- Canonical ticket frontier and machine ledger live in the LocalAI integration
  lane: Spec0005, tickets0162–0182, plan0015 execution JSON. Native171/173/178
  and mixed167/handoff168 remain open. Tickets0180–0182 are complete for their source/diagnosis scopes. Native release
  remains unqualified after actual bounded follow-ups.
- Eight cumulative recording-stage attempts are retained. Real public AMI DIA
  returned12turns/3speaker vectors with measured DER0.1530782443. The latest
  diagnostic case captured a Z leader with a same-process D thread waiting in
  dxgvmb_send_sync_msg after signal requests; owned numeric exit remained missing.
  App15s release and Windows held-handle wait failed. Later absence is quiescence
  evidence only; original failures persist and host-side cause remains unproved.
- A proposed <=15minute idle-only Speech maintenance/control/restore window is
  prepared outside Git. The operator subsequently approved this ONE idle-only control/restore window;
  its independent restore timer was armed before the pause. Broader service pauses,
  GPU reset/reboot and automatic follow-up cases remain excluded.
- Requested final broad review and post-review remediation remain0/1 each,
  reserved until the ticket acceptance frontier actually closes.

## Accepted scheduling and residency policy

| Order | Workload | Historical work may outrank it through aging? |
| --- | --- | --- |
|1 | Live transcription and live text-to-speech | No |
|2 | Interactive search/retrieval | No |
|3 | Recent voice-recording transcription | No |
|4 | Historical retranscription/diarization | Not applicable |

An old recording newly discovered/synced is historical. Consumers declare an
authorized class; they do not grant themselves effective priority. Protect active
sessions and pending interactive demand automatically. Idle cached residency is a
warm-readiness preference and may be reclaimed according to quiet-period policy.

Quiet eligibility follows 15 minutes without interactive service activity, no
active protected session and no pending protected demand. This measures service activity, not human presence.
Overnight alone never overrides active demand. Manual controls can change quiet/
awake preferences. Returning interactive work immediately outranks historical work;
report any cold model loading and prewarm when advance notice is available.

Temporary overrides have a visible operator-chosen expiry. A service restart does
not extend that expiry. Persistent overrides require explicit selection. Retain
actor, scope, reason, creation/expiry, effective policy and revocation audit metadata
without raw request/payload logging. Apply controls under trusted operator authority.
Scheduling overrides never bypass resource safety or imply permission to terminate
an unrelated active session.

Initially admit at most one historical GPU stage at a time. Coexistence with
protected workloads requires qualified free capacity and responsiveness. Historical
work can remain starved for hours; explain the wait and resume automatically when
eligible. Do not use starvation aging to bypass this accepted precedence.

Borrow HPC backfill and QoS concepts: explicit resources/walltime, visible pending
reasons, separate priority/preemption/limits, and checkpoint/requeue where safe.
Admit backfill only under declared resource/time assumptions that preserve higher
priority demand; unexpected return must still trigger the agreed handoff policy.
Process suspension alone is not memory-release proof.

## Work-unit and output contract

Transcribe Audio selects immutable recording inputs, stages authorized bounded
media, retains source/conversation identities and owns stable unit/request IDs,
recording checkpoints, timeline offsets and canonical export/store updates. The
Shared GPU service owns admission, loading, execution custody, residency, cancellation
and observed resource release for every GPU stage. No hidden direct-GPU branch.

Unit size is chosen from measured processing/yield cost and model-quality needs,
not from the assumption that a fixed audio duration is a fixed execution duration.
Completed units survive restart/preemption. An unfinished unit may be discarded
only after its outcome/custody is reconciled; retry cannot duplicate an unknown
running request. Retain raw/intermediate results and explicit unresolved attribution.

Reconcile anonymous speakers across the recording before final transcript publication.
Equal numeric labels from independent units are not the same speaker. Preserve
ambiguous cases rather than forcing false continuity. Qualify continuity through
interruption and chunk boundaries, including overlap/deduplication and timestamp
alignment. A recording remains one recording across its work units.

Publish the complete canonical generation only after required stages succeed and
integrity checks pass. Preserve original media and prior transcript versions with
lineage. Do not mark undiarized work complete, duplicate exports on uncertain effects,
or silently recompute speaker identities/accepted contextual claims.

## Qualification sequence and ownership

| Packet | Owner | Deliverable / exit evidence |
| --- | --- | --- |
| A: freeze contracts and profile evidence | Joint coordinator | Exact source/runtime/model identities, private cohort manifest, profile limits, input controls, direct baselines and prospective thresholds |
| B: shared policy and custody | LocalAI | Correct precedence, quiet/protection/override behavior, managed native adapters, resource/exit evidence and deterministic contract tests |
| C: recording workflow | Transcribe Audio | Durable units/checkpoints, stable IDs, recording-wide speaker reconciliation, original metadata and guarded exports |
| D: joined source qualification | Joint coordinator | Authenticated integration/recovery fixtures, faults and source checks with no model effects |
| E: bounded native comparison | Joint coordinator | Matched direct/managed real operations, cold/warm and interference evidence, independent cleanup and quality verdicts |
| F: representative useful-work pilot | Transcribe Audio with LocalAI | Verified canonical artifacts/library, operator-reviewable quality evidence, mixed-load response and resumability report |

A precedes B/C. B and C can proceed independently only after shared contracts are
frozen; D joins them before E/F. First verify that direct model profiles can meet
their own quality/feature needs; a native model miss is not automatically a manager
regression. Conversely, useful output alone does not prove shared management safe.
Protected profiles need their actual declared capabilities: complete-WAV TTS cannot
stand in for first-audio streaming, and bulk ASR cannot stand in for live ASR.
If such a dependency is unqualified or unavailable, report that dependency rather
than claiming the mixed-profile pilot passed or silently expanding its model scope.

Before model execution, derive one bounded packet from this plan with exact model
assets/adapters, GPU/CPU/RAM reservations, fresh capacity checks, deadlines, maximum
attempts and cumulative wall-time allowance, input/actor scope, output/evidence
paths, cancellation/cleanup identity and semantic exit. Validate the allowance covers
baseline, managed comparison and cleanup. Preserve existing consumed allowances and
review/TDD counters; only a prospectively authorized envelope may run native proof.
Do not turn packet boundaries into repeated approval requests for ordinary work
already authorized by a later execution instruction.

## Metrics and prospective evidence freeze

Warm mixed-load goals: embedding response p95<=500 ms, TTS first audio p95<=1 second,
live-ASR response p95<=2 seconds, under a declared feasible offered load. Freeze exact
request/first-useful-output timing definitions and sample counts before trials.
Include queuing and scheduler effects. Report every cold start, worst-case latency,
interruption/reload and foreground-return delay separately; never hide a cold
transition caused by historical work to improve a warm score.

Measure native profile experience and manager-added cost against matched direct
execution with the same inputs, controls and assets. Direct baselines are bounded,
owned qualification operations with explicit capacity/cleanup proof, never a
production fallback. Do not reclassify expected-warm candidate arrivals into an
excluded cold bucket when historical interference caused the transition. Freeze the manager-overhead,
throughput, cold-return and yield/release limits from retained applicable contracts
and usable baseline evidence before candidate execution. Preserve old stricter
failed gates independently; a new consumer goal does not erase them. Historical
whole-file and faster-whisper numbers are not portable acceptance for this pipeline.

Use a small declared cohort spanning applicable short/long, single/multiple-speaker,
noisy/overlapping and interrupted cases. Review reference audio excerpts for word
accuracy, speaker assignment and timestamps. Freeze the scoring rubric, coverage,
error tolerances and review method prospectively. Old transcripts, provider agreement
and speaker-count equality are comparison inputs, not automatic ground truth.

## Acceptance Criteria

1. Every executed GPU stage has one manager-owned lifecycle/reservation and verified
   release; stale/missing capacity never admits unsafe work. No unmanaged fallback.
2. Accepted precedence, no historical aging promotion, automatic protection,
   quiet entry/exit and scoped expiring/persistent overrides work across restarts.
3. Warm goals and separately frozen cold/yield limits are evaluated honestly under
   mixed load; protected live sessions retain their required streaming behavior.
4. Long-recording progress survives bounded interruption/restart without lost or
   duplicated units, exports or canonical commits. Unknown outcomes remain explicit.
5. Completed pilot transcripts have reviewed words/speaker/timing quality and
   recording-wide anonymous speaker continuity, or explicit unresolved evidence.
6. Original recording/conversation identity, calendar/provenance, media and earlier
   transcript versions remain intact; DOCX/TXT/JSON/library results agree.
7. Exact source/installed runtime identities and all pass/fail receipts are retained;
   separate management, model-profile and consumer acceptance verdicts are issued.
8. Voice AssemblyAI operation remains unchanged; historical bulk activation remains
   a subsequent rollout decision, not an incidental result of running the pilot.

## Recovery, limits and rollout

Reuse existing stable-ID submission, known-rejection backoff, observe-only unknown
recovery and guarded canonical-output semantics. Tune finite retry/backoff limits
inside each packet; isolate failed work and continue unrelated safe work. Do not
retry ambiguous submissions merely because observation is slow. GPU custody or
telemetry failure blocks affected admissions until reconciled, not by dropping
reservations or falling back around management.

Pause historical admissions on resource/custody failure, unsafe telemetry, repeated
protected-work response regression or canonical-output integrity failure. Preserve
active foreground work and evidence; qualify recovery before resuming affected work.
Report normal priority waiting separately from a fault or failed qualification.

This milestone ends at a bounded useful-work pilot report. Any broader canary or
full backlog rollout needs its own bounded cohort/stop criteria under the then-current
operator execution scope. No automatic deployment or full backlog run follows this
PLANNED document. A failing candidate changes tactics/profile prospectively rather
than relaxing gates or looping without a remaining allowance.

## Validation

Run the changed surfaces' local deterministic scheduling, lease/override expiry,
protected return, capacity, cancellation/quiescence, protocol, stage handoff,
checkpoint, cross-unit speaker/timestamp and export/library checks. Exercise restart
and lost-acknowledgement cases at public seams. Run short real-media CLI/export
smokes within the native packet. Reopen final exports and verify library/provenance.

Retain commands, results, source commit and installed identities, selected test
cohort, environment differences, native request/process IDs and terminal receipts.
Hosted GitHub CI is skipped under operator policy; do not buy capacity or weaken
separate execution/deployment gates. Current drafting validation is documentation
consistency/links/whitespace only, not software/model qualification.

Preserve LocalAI's reserved final broad review and TDD remediation sequence; this
interview's read-only fact checks do not consume or substitute for that acceptance.

## Implementation checkpoint |2026-11-11

Operator confirmed shared understanding and invoked to-spec/to-tickets/implement-spec
with delegated frontier execution. Canonical implementation tracker is LocalAI
qualification spec0005 and tickets0162-0169; execution ledger is
qualification/docs/dev/plans/0015-2026-11-11-managed-diarized-execution.json.
One integration lane spans the two independent Git repositories on branches named
codex/managed-diarized-0095; production source checkouts remain unchanged.

Source tickets0162/0163/0164/0165/0166/0169 are closed from integrated public checks.
The managed CLI performs CPU staging, real authenticated client calls, durable
recording checkpoints and guarded versioned exports; source tests use scripted
external engines and do not qualify CUDA or model quality. The existing AssemblyAI
route and paused screen watcher remain unchanged. Native0167 and handoff0168 remain
blocked: a qualified native streaming-TTS cohort is not established, the retained
live-ASR reservation/headroom exceeds current free capacity, and a reviewed
multi-speaker/timing cohort remains to be frozen. A scope choice on upstream
prerequisite qualification was asked asynchronously; silence is not authorization.

Requested final broad review and post-review TDD counters remain0/1 and0/1, pending
closure of the requested ticket set. Feature-level red/green tests are implementation
evidence and do not consume that final sequence. Hosted CI and model execution were
not used for source acceptance. No deployment or broader backlog activation occurred.
