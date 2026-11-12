# Managed diarized recording qualification and integration

State: OPEN
Owner: primary integration coordinator
Scope: qualify a bounded resumable local transcription-and-diarization pilot
Execution: bounded recording stages accepted; remaining live-ASR, managed embedding and mixed-workload acceptance open
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

- Ticket0173 is complete for bounded native recording stages: cumulative13
  recording attempts retain all historical failures. DIA12 returned12turns and
  three256D vectors; ASR13 returned121timestamped words on the same59s public
  AMI clip. Stop-owned completed in2.748s/2.741s with actual numeric KILL exit-9,
  reservation0, model/process/cgroup absence and both outer cleanup proofs.
- Independent measurements are WER11.81% and DER15.31%; timestamp MAE remains
  undefined with two missing reference boundaries. These are measured-only
  short-cohort results, not full mixed-pipeline or human word-timing acceptance.
- Ticket0171 remains open. A direct20warm live-ASR cohort passed p95.990s;
  managed readiness hit its80s bound before streams. A separate bounded diagnostic
  retained the failure without locating its phase. A CPU-masked import probe
  completed, but cannot establish the original native delay's cause.
- Ticket0178 remains open. Direct BGE-M3 passed20/20 retrieval matches and warm
  p95.0678s at its2GiB cap. Managed work stayed queued without a worker because
  the qualification helper compared monotonic capacity with wall-clock time.
  The narrow clock-domain fix passed a real-worker/HTTP red-green regression and
  eleven focused checks. A newly pinned managed-only candidate awaits admission;
  the old incomplete pair and conservative three attempt intents are retained.
- Current-epoch capacity qualification retained260 unique strict-valid samples
  with120s minima on Windows monotonic, acquisition UTC and Linux clocks, and
  exact cleanup. The previous Linux-only-duration cohort remains rejected.
- Source baseline359checks/49optional-live skips and70joined app checks remain
  historical evidence. The bounded ASR/pilot harness update passed7focused tests;
  the consumed twelve-start packet still refuses execution. Real local ASR CLI
  output exported to matching DOCX/TXT (eight paragraphs), without publication.
- Mixed167 and handoff168 remain open. Final broad review and subsequent TDD are
  still0/1 each. AssemblyAI routing stays active and the historical backlog stays
  disabled. No new Speech pause, GPU reset, host reboot or driver change occurred.

## Earlier implementation and recovery checkpoints (preserved)

- Isolated integration branch `codex/managed-diarized-0095` contains the historical
  precedence/quiet policy, owned stage workers, resumable processor, guarded CLI,
  recording-wide speaker reconciliation and qualification/evidence tooling.
- LocalAI source regression:359tests OK with49optional/live skips; joined app70
  tests pass. These are source checks, not native acceptance. Later focused
  controller/provider changes have separately retained red/green evidence.
- Native CPU streaming Phonon and public AMI reference prerequisites are complete.
  Optional CPU BGE strategy was measured and rejected for the unchanged500ms
  target; GPU embedding candidate remains unmeasured after one interrupted load.
- Real Nemotron3GiB canary produced an incremental transcript, but original5s
  capacity acquisition contract failed under load before a20warm paired cohort.
  Full paired ASR and combined useful-output acceptance remain open.
- Diagnostic sampler progress has CPU/provider proof that fresh diagnostic data
  cannot refresh stale GPU counters. Isolated cached-factory0180 qualification
  achieved240unique valid idle samples. Subsequent DIA runs maintained fresh
  acquisition under the accepted age/headroom gates.
- Production voice/main watchers were freshly verified active, selected jobs
  Assembly-only and screen processing disabled. The approved Speech maintenance
  window completed and was restored as noted below. Historical negatives remain.
- Canonical ticket frontier and machine ledger live in the LocalAI integration
  lane: Spec0005, tickets0162–0184, plan0015 execution JSON. Native171/173/178
  and mixed167/handoff168 remain open. Tickets0180–0184 are complete for their source/diagnosis scopes. Native release
  remains unqualified after actual bounded follow-ups.
- Eleven cumulative recording-stage attempts are retained. Real public AMI DIA
  returned12turns/3speaker vectors with measured DER0.1530782443. The latest
  diagnostic case captured a Z leader with a same-process D thread waiting in
  dxgvmb_send_sync_msg after signal requests; owned numeric exit remained missing.
  App15s release failed; latest outer Linux and Windows cleanup passed. Earlier
  failed held-handle waits remain retained. Later absence is quiescence evidence
  only; original failures persist and host-side cause remains unproved.
- The approved idle-only Speech maintenance/control/restore window completed in
  402.534s of its900s allowance. The same release failure persisted with Speech
  paused. Speech and both previously ready backends were restored and verified
  before retiring its independent timer; AssemblyAI watchers stayed active.
- Graceful worker cleanup and EOF-assisted stop are source-validated, but actual
  native release still failed. Early EOF records rolled out of the bounded
  diagnostic history, so EOF delivery is not claimed from registration alone.
- The operator approved one NVIDIA-only restart window. An independent Windows
  Speech restore task was armed before pausing Speech. The single restart task
  hit its 120-second limit without a terminal operation receipt: reset outcome
  remains unknown and no retry occurred. Device status subsequently read OK,
  but the UUID query and recovery-task readback timed out. Speech is active and
  Pocket ready; Nemotron is stopped/blocked by unavailable GPU telemetry.
  AssemblyAI watchers remain active. The restore task is retained; its scheduled
  trigger has passed but completion is unverified. No qualification control ran.
  WSL shutdown, Windows reboot and driver updates remain outside approved scope.
  Native acceptance and final review remain open; application variants are held.
- Continuation readback verified Speech/Pocket/Nemotron ready and both AssemblyAI
  watchers active. The Windows restore task returned0. The current host epoch
  differs from the failed window; exact GPU and driver match, with a fresh DXCore
  binding. Earlier reset completion remains unknown. Missing temporary integration
  checkouts were restored from retained commits, and runtime pins match. One new
  model-free source qualification precedes the unchanged bounded DIA control;
  the recording ledger retains11 physical starts with a12-start ceiling.
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

## Prospective recording continuation after the recovered DIA gate

The twelfth recording-stage start passed real DIA output and bounded release.
The original twelve-start packet is consumed and remains immutable. The next
prospective packet permits one ASR-only start, cumulative13, retaining all12
prior entries. A later separately admitted two-stage pilot may use cumulative15
only after predecessor acceptance. The harness validator supports this bounded
ceiling; it does not replenish any old packet or authorize automatic retries.
The native runtime, cleanup deadline, allocator caps, model math and freshness/
headroom gates remain unchanged. A CPU regression check proves the consumed
original packet still refuses execution without changing its ledger.

This advances trustworthy transcript/diarization from isolated implementation
toward bounded real-artifact shadow evidence. It does not establish automatic
production processing, representative quality, named identity or knowledge reuse.

Local harness validation: `PYTHONPATH=/tmp/localai-plan0095-integration/qualification
python -m pytest -q tests/test_managed_recording_native.py` passed7 tests. Red
receipt caught the old13-load rejection; the first green attempt exposed a
missing required path in the new test fixture, corrected without further runtime
changes. All receipts remain under private `plan0095/native0173/asr-successor-*`.
This ordinary implementation TDD does not consume the reserved post-review pass.

## Current continuation checkpoint — 2026-11-12

This section supersedes earlier open-prerequisite descriptions above. LocalAI
0173 recording stages,0178 standalone CUDA embedding,0186 controller observation
and0187 prospective controlled-stop evidence are complete within their documented
scope. Real recording ASR13 and DIA12 emitted independently scored outputs and
released owned resources. They do not establish representative quality or mixed
acceptance. Missing timing boundaries remain unresolved.

LocalAI integration `1f4f730a3` adds the one-clip public CLI composition: actual
CLI, authenticated service, stage exit/checkpoint handoff, DOCX/TXT/JSON library
readback and replay are CPU-validated. The native packet template is explicitly
non-executable pending bindings and admission; no pilot starts14/15 occurred.

0171 remains blocked:10/10 bounded starts consumed. The final diagnostic reached
readiness in65.484s but submitted no audio; the managed warm cohort remains absent.
Previous80/90s cold failures remain negative evidence. No further load is admitted.
0167 still requires that cohort, reviewed quality/timing criteria, paired recording
baseline, mixed scenarios and four-clip/recovery coverage.0168 stays open. Final
review and post-review remediation counters remain0/0; no final sequence was run.

Primary joined source checks at LocalAI1f4f730a3:12 tests passed17.318s using
Linuxbrew PATH. Initial missingffmpeg PATH failure was environmental and retained;
no product workaround or model changes were made. Hosted CI was not run.
Production Speech and both Assembly watchers remain active; the screen backlog
was not activated. Earlier NVIDIA reset outcome remains unknown.

## Authorized continuation — 2026-11-12 morning

Operator `ok go` accepts the proposed same-model parity quality gate and fresh
bounded live-ASR cohort preparation/execution within unchanged resource/deadline
limits. Absolute word/speaker errors and unresolved timing gaps remain separate;
parity is not representative quality or a fabricated numeric timing pass.
Preserve all ten live-ASR starts and their failures; freeze new cumulative limits
and baseline comparability before effects. No production interruption, reset,
model download, routing or historical-backlog activation follows from this.

Linux boot changed to827d9db9; both temporary integration checkouts disappeared.
Restored existing registered worktrees from retained app4ca52a7/core375ce3a72;
no main changes or pruning. Current-epoch capacity source requalification must
precede model admission. Root owns native serialization/admission; delegated
tracks are source preparation, cohort preparation and quality-policy source work.

CPU-only public `prepare_media` preflight at the exact retained commits reproduced
source/decoded/unit SHA256b56bd92600d2a6047a0967c0b064e873f0cecde005fbd413d322b1549aa910fc
and59s duration. Private receipt: user-state Plan0095/native0167/preparation/
resume827d-cpu-preflight/receipt.json. No model or collector started by this check.
This continues the level2 bounded shadow objective; target is trustworthy useful
artifacts under managed priority, measured by real output/parity and resource
release evidence. Production automatic processing remains outside this packet.

## Prospective DIA-only recovery after observed V2 cancellation

The failed160s whole-CLI case remains failed. Its ASR checkpoint contains121
words; saved service-state and job database independently retain DIA cancelled
and no resident models. Existing public retry can explicitly advance the cancelled
DIA attempt while preserving ASR. This is the planned restart/recovery milestone,
not another full-pipeline replacement or an automatic retry of the old packet.

Define ONE prospective cumulative recording start16, DIA-only. Preserve the full
15-entry ledger and immutable snapshots of original checkpoint/service evidence
before application state updates. Original max15 packet stays exhausted; max16
is accepted only with `allowed_stages=['diarization']` and exactly15 retained
entries. ASR is rejected before ledger mutation or capacity query. No source case
or GPU load follows until CPU recovery proof, exact packet/pins, demand/capacity,
owned cleanup and primary admission pass. Existing160/170/15/4CPU12GiB and strict
5s/3.5GiBfloor constraints remain; no whole-window extension or model change.

Seams tested: public `native_limits` prospective scope and `worker_exec` custody
refusal. Red rejected16; green permits scopedDIA16 but rejectsASR and consumed
old15. A second red caught an external capacity-query boundary before validating
retained15history (external process mocked; no NVIDIA call); green refuses there
first. All9 targeted harness tests pass0.09s. Red/green receipts are private
`native0167/preparation/recovery-{custody,history}-*`. This is ordinary implementation
TDD; reserved final review/post-review counts remain0/0.

## Prospective staged mixed custody — source preparation only

The next independently useful mixed batch is M1 historicalDIA preemption/cold
interactive return, B2 matched direct interference and M3 managed recent-voiceDIA
interference. Currentpolicy vetoes historicalwork on any activeinteractive demand;
M3 must therefore state recent-voice scope. The exact frozen offeredorder is
DIAcoldprobe first, then BGE/TTS coldprobes, then two furtherDIA jobs spanning the
20fixed1Hz warm arrivals. No all-order/co-residency orkernel-overlap claim follows.
Live-ASR/fourclip remain separate. CurrentTTS sourceconfig is the actual170
Phonon CPUq8 profile (defaultvoice/threads2); stale private paths are repinned
explicitly with unchangedengine/assets/math, producing a truthfulnewidentity.

A NEW prospective DIA-only cohort may retain16originalrecordingstarts and allow
at most3new starts, cumulative19. This does not renew the consumed max16recovery.
The harness accepts19 only with `allowed_stages=['diarization']`,
`budget_contract='staged-mixed-dia-v1'`, `retained_physical_starts=16` and the exact
canonical JSON SHA256 of all16retainedentries. Before capacity/effects it verifies
that prefix on every load; ASR, missingcontrols, modifiedhistory and exhausted
oldbudgets refuse. The percase one-load gate and independently frozen three-case
order remain mandatory. Corresponding proposed BGE/TTS modelstart ceilings7/5
retain their original4/2intent histories. No nativecase is admitted by this source
change; source191currently has a retainedlate-arrival CPUfailure under diagnosis.
All160/170/15,4CPU12GiB,5sfreshness and3.5GiBfloor limits stayunchanged.

Publiccustody red rejectedscoped19; green10harness tests pass0.15s. The new test
also proves changedretainedhistory stopsbeforeanexternalcapacityquery, ASR is
refused and the original16packet stays exhausted withoutledgerchanges. Receipts:
private `native0191/preparation/mixed-custody-{red,green}.txt`. Ordinary featureTDD
leaves finalreview/postreview0/0.

## New190 live-ASR prerequisite accepted |2026-11-12

0171 is now complete for the explicitly new fingerprint
a0b458f771ee539f2c4e678d19af870e6cd912e1b9c72f55da06d12be46e1a02:
verified preextracted CPU staging, unchanged model/math/PCM/context/cadence/3GiB,
and deny-python-network-v1. Source0190 is complete; its installed-SDK config-only
probe passed without model construction. This advances the existing level2
bounded useful-work qualification target; production activation remains separate.

All four prospectively frozen DIRECT1,DIRECT2,MANAGED1,MANAGED2 cases passed.
Retained physical liveASR15/15;40warm streams plus four probes succeeded withWER.1.
Nearest-rank pooled20 p95: direct.981272771s, managed1.067927542s. Delta.086654771s
passes the frozen.098127277s budget; absolute2s passes. Separate cold times were
61.132/53.571/63.936/54.852s; yields.845/.955/.849/.970s. Numeric exits-15/stopped,
managedreservation0/models{}, allocatorpeak2,663,251,456B<3GiB, strictages<.294s,
loadedfloor>3.5GiB, exactknownprocess/cgroup/Windows/source/tunnelcleanup pass.
Parent independently recomputed rawframes/ACKs/partials/p95/delta and checked
fresh process absence. ProtectedSpeech demand remained empty throughout allfour.

Private aggregate `native0171/newvariant190-paired-comparison.json` SHA256
790536996ffd6ba181738269dcec2fc9838903ffed154edec8e3354e6470e1a1;
parent readback `newvariant190-primary-paired-readback.json`. This is a two-
generation-per-route syntheticstreaming qualification, not a singleworker20
claim or causal cold-speed improvement. All old failures stay retained.

0191 source queued-offers V2 fixes are integrated at coref2f70365a; primaryfull
local suite passed384tests with49optional/live skips in249.608s. Its retained
ResourceWarnings are not a warning-free claim. NativeM1/B2/M3 and19/7/5 custody remain held pending exact
source/pin/individual admission. Source0192 prepares fourclip paired quality
reporting, preserving missing captures, null timing and final-reconciliation gaps.
0192 source/evidence is complete at core7a5cd6e07: primary22joinedtestsPASS1.105s;
independent actualfourclip report remainsBLOCKED withonehistoricalscoped milestone,
3missingcaptures andlongclipminimumfourunits. Ledgercheckpoint core2ee79272c.
The operator has been asked for missing recording performance limits and human
word-timing scope; those dependent gates stay open. 0167/0168 and final0/0 remain.

## Retained M1 pre-execution failure and canonical custody repair

The first frozen mixed M1 v3 failed before a model exec: its preparation supplied
raw ledger SHA958e981a... where the guard requires canonicalfirst16 SHA56415edf....
Actual owned wrapper exited1; the harness waited until its original160s deadline.
No physical exec counter advanced: recording16/BGE4/TTS2 remain unchanged. Manager0,
worker/unit/cgroup and Windows/source/tunnel cleanup passed; parent independently
checked knownPIDs/cgroup absence and unchanged ledger bytes. B2/M3 v3 stay held
and will never execute. Failed packets/reports remain intact; existing Gitrefa134905
and a pinned-identical private copy retain original appsource bytes after edits.

Source0194 exposes `mixed_retained_history_sha256(attempts)` and uses that same
helper in the existing guard. Its public roundtrip test proves raw-byte hash
rejection before capacity, correct prefix reaching a raising capacity fence before
mutation, unchanged temporary ledger and stable first16 identity after appends.
Primary joined75testsPASS14.69s with documentedLinuxbrewPATH/qualificationPYTHONPATH.
An initial command withoutPYTHONPATH had3importerrors/8passes and is retained;
it is not a product failure. Private preparation also calls the actual shared
helper; no duplicate digest algorithm or GPU retry occurs in source validation.
Source0193 separately addresses cold worker exit observability. Any corrected
successor needs new frozen hashes and individual admission, with originalphysical
ceilings19/7/5 unchanged. No previous failed packet is promoted; final0/0 remains.

## V4 response failure and status-copy source repair

Corrected V4 M1 completed all40warm requests in121.625s and proved historical
DIA cancellation/release in2.769s. It still failed caller-visible targets:
BGEp953.29632646s>.5, firstTTSaudio2.984742031s>1. All20offers/role were within
9ms of their fixed1Hz schedule; no failed or late sample was discarded. Pipe
first-audio timestamps and caller observation remain distinct; no retrospective
SLO pass follows from earlier pipe arrival. B2/M3 V4 are stopped/unexecuted.

Counters are recording17/BGE5/TTS3, preserving first16 canonical recording
history and original BGE/TTS evidence. Actual exits DIA-9/BGE-15/TTS-15; Manager0,
allknownPIDs/recordedPGIDs/cgroup and Windows/source/tunnel cleanup pass. Parent
independently recomputed p95 and checked fresh absence. Private evaluation SHA
39e0aafec163792783107bd8b6637e58c6c2402cdd156313c07b930b181639ed.

Source0195 reproduced one avoidable cost with public CPU process/HTTP boundaries:
20retained audio results caused171MB of payload copying during status cycles.
Opt-in metadata observation now supplies event count without copying events;
default event/output/result/cursor/replay/history behavior remains. CPU after-case
copied audio only for explicit output; this does not attribute the entire native
failure or prove native speed. Core40b0eecba primary full392testsOK49optional/live
skips251.200s; application75PASS15.71s. Known fixture ResourceWarning retained.

A source195 SINGLEM1 proposal may use17/5/3->18/6/4 under unchanged19/7/5 ceilings,
with an explicit one-case guard binding actual consumed V4 history and refusing
replay. TTS control-source identity must be recomputed; oldf149 acceptance does
not transfer automatically. PairedB2/M3/performance limits remain held pending
operator input; onlyone remaining load/role after that singlecase cannot cover
both paired routes. No new budget, route activation or final review occurs here.

## Independent client measurement successor |2026-11-12

Source197's last bounded diagnostic consumed19/7/5 and closed with complete trace
and owned cleanup. It retained failed metrics, never qualified a neural SLO.
The trace proved that one receiver spent4.935s handling unrelated HTTP requests
before another result GET. Source198 replaces that coupling prospectively with
INDEPENDENT_CLIENT_RECEIVERS_V3: at most40independent request receivers, unchanged
fixed20offers/role/1Hz/300ms tolerance and one physical worker/slot. Direct FIFO
retains actual per-job output before reuse; actual client delivery uses NOW.
Incremental cursors preserve every frame. Old V2 failures are not rescored.

Primary full core394testsOK49optional/live skips271.866s at e2a39138d. The new
V3 source is validated, while native V3 is not run. Under the existing approved
implementation/qualification goal, a new bounded successor is being prepared:
fixed M1,B2,M3, stopfirstfailure, at most3new DIA/BGE/CPU-TTS starts, cumulative
22/10/8 with all19/7/5spent histories retained. Old exhausted packets stay held.
No source change alone admits a native effect; exact packet/pins/demand/capacity/
individual primary admission remain mandatory. ASR/live0, unchanged assets,
160/170/15,4CPU12GiB, strict5/floor3.5 and original500ms/1s targets remain.

Source199 prepares only scoped22: explicit V3 layout/contract, DIAonly, exact
canonicalfirst19 history via the shared preparation helper. Unscoped20/21 and
missing/wrong controls reject; ASR rejects; changed retained19 stops before an
external capacity call; correct prefix reaches a raising CPU capacity fence
without mutation. Original19packet stays exhausted. Primary12guardtestsPASS.16s
and joined76application testsPASS11.02s. This is explicit prospective allowance
preparation after the evidenced benchmark fix, not a reset or automatic retry.
Native cohort remains held until final source/history/pin checks and admission.
Full0167/0168 and missing recording-performance/word-timing decisions remain open.


Independent-client V3 M1 subsequently completed107.231s; all40warmrequests
succeeded, but primary recomputed BGEp952.208080383s/TTS2.144715370s both FAIL.
Ticket0200 cohort STOP: B2/M3unexecuted, no replay/replacement despite unused
ceiling headroom. Actual starts20recording/8BGE/6TTS; old histories unchanged.
All42actors stopped, actualDIA-9/BGE-15/TTS-15, Manager0/models{}, outer source
heldexit1/jobempty/tunnelabsent and nativecgroup absent; primary freshread agrees.
Root protectedmonitor54polls finished without demand/abort. No production switch,
backlog activation or qualification claim. Bounded public CPU delivery diagnosis
0201 follows observed high service/client intervals; it grants no native budget.
Recording acceptance inputs remain pending; finalreview/remediation0/0.


## Missing native boundary timing | prospective diagnostic204

CPU201 complete20.857s/40rows/.372sBGE/.357sTTS did not reproduce native delay;
no source fix or second CPU capture. Raw trace complete45,527rows/7,057,566bytes,
actual4CPU/12GiB/child-15/cgroupabsence, analysisf31a5771... retained privately.
Native/CPU vector serialization sizes differ; no causal claim follows.

Under the standing implementation/qualification goal,202/203 prepare missing
per-request parent-pipe/service-reply timing and exact one-shot custody for204.
This is a NEW diagnostic purpose, prospective baseline20recording/8BGE/6TTS to
ceilings21/9/7/maxone per role, oneM1only, no replay/replacement/B2/M3 or sample
transfer. It is never authorized merely by unused old22/10/8headroom. OldV3
cohort remains STOP. Native HELD until exact source/trace/custody proof/pins,
currentboot/capacity/protected-demand and individualprimary admission. Original
160/170/15/4CPU12GiB/strict5/floor3.5/20offers1Hz/.300 bounds remain. Instrumented
results NEVER qualify neural SLOs even if measuredtargets pass; terminalsuccess
is attributable missing timing+actualcleanup, not another acceptance attempt.
NoASR/live/prodroute/backlog/download/reset/pause. Finalreview/remediation0/0.


Source204 diagnostic completed97.018s/40warm/actual21recording9BGE7TTS; numeric
DIA-9/BGE-15/TTS-15/42actors/Manager0 plus fullsource/cgroup/knownPIDcleanupPASS.
Root protectedmonitor49polls finished/no demand. TraceINCOMPLETE: actualBGE/DIA
codecNone missed by the all-codec CPUproof; onlyTTS parentreceipt/perPIDCPU mapped.
Used artifacts remain immutable; no retry or neural/SLO promotion. Instrumented
3.228sBGE/3.642sTTS values are diagnostic only. Source205 repairsprivatecoverage
with actualunloadedfactoryNone/None/TimedSpeechCodec inventory beforemixedCPUproof.

Partial204 evidence identifies accept-loop maintenance as a candidate:312ticks
occupied21.8/24s while responsewrites were milliseconds/pre-servicemethod waits
.755–.877s. Native TTS frames span.934s versusCPU201burst.0055s. Source206 is
CPUONLY: one matchedstreaming pre-fixcapture and publicHTTPcausalRED, then only
ifproved a bounded service_http maintenance-isolation fix + ordinaryfeatureTDD.
No nativepacket/budget/production/backlog/pause/reset/download. Full0167/0168 and
recordinglimits/wordtimingdecisions remain open; finalreview/remediation0/0.


## V4 source preparation | no native admission

Source206 isolates maintenance in oneownednonoverlapactor from HTTPacceptance;
publiccausalRED/GREEN and19localchecks pass. MatchedCPU connectionadmission p95
.245s→.0015s, BGE.8088→.5144 STILLFAIL.5, TTS.8497→.5219PASS1 (notneuralproof).
Source207 publicHTTP503metadataRED1.930s→healthydataGREEN2tests21.148s removes
unrelated status polls: result/state forBGE, output/state plus terminal result
forTTS. ExplicitINDEPENDENT_CLIENT_RECEIVERS_V4/schema4/ROLE_DATA_ENDPOINTS_V1;
oldV3packetsrejectbeforefactory. Arrivals/queues/frames/modelmath/targets unchanged.
One matchedCPUfeedback and finalcombinedcore/appchecks remain; no nativepass.

208preparesONLYscoped24recordingguard withactualfirst21canonicalhelper, V4/data
protocol andDIAonly in a NEWapplicationchildworktree. Primaryapp11scriptpath
stays unchanged for oldhistory validation. 209prepares NEWfixedM1/B2/M3 proposal
fromactual21/9/7 to24/12/10, max3newstarts/role, stopfirstfailure/no replay;
alloldfailedcohorts andconsumed204diagnosticremainclosed. 210isPLANNED/HELD,
notadmitted bythis plan: needsallsourceproofs/feedback/fullchecks andeach exact
primaryfreeze/admission/currentboot/capacity/protected-demand gate. NoASR/live,
no tracer forqualification, sameassets/math/16017015/4CPU12G/strict5/floor3.5.
No production/backlog/pause/reset/download. Full0167/0168/recordingperformance/
humanwordtimingremainopen; finalreview/remediation0/0.


## V4 terminal evidence | 2026-11-12

Core source9007125a7 passes401 local unittest checks with49 optional/live skips
in284.481s; application guard child2d9ecce065 passes81 selected pytest checks
in11.47s against that core. Primary application script remains11c3fc8c for
historical evidence validation. No installed runtime change follows these checks.

Uninstrumented native0210 M1 completed40 correct warm requests in104.049s,
but BGE p95 1.065290438s exceeds0.5s and TTS firstaudio1.068572772s exceeds1s.
Verdict FAIL_STOP; B2/M3 remain unexecuted and unfrozen. Actual cumulative
recording/BGE/TTS starts22/10/8. No unused allowance authorizes a retry.
Independent evaluation43dd283118564e6533602a494fd663303a82c381ad457c9da2147aa8d0ee9fcf
and custody e2fb90d80dc0b1d1b9e4ad473c5fbdf0005f7c709bdbcfccd76950e1c8074993
are retained under user-scoped plan0095/native0210. All known owned processes,
groups, cgroup, tunnel and source exited; Manager0 and42 actors closed. Root
protected-demand monitor finished56 polls without protected demand.

Core ticket0211 audits native/CPU fidelity read-only;0212 prepares private
all-role/HTTP boundary markers with small CPU proofs. Neither permits another
native run. Full0167/0168 acceptance remains open, including recording performance
and human word-timing criteria. Reserved final broad review/TDD remain0/0.
These steps advance trustworthy diarized processing from built toward shadow
maturity; foreground timing is still unqualified. No production/backlog activation.


## CPU isolation frontier | 2026-11-12

Read-only211 audit closed: retained native/CPU job stores both44jobs/~7.79MB;
gross payloadgrowth unsupported. Small213sleep/CPU pair verifiedrealCPUdemand
withoutreproducing nativefailure; originalzero-start setuperror retained.
Full214ONE20perrole pair reproducedsearchp95.751007146/.750849972FAIL.5
whilefirstaudio.775913622/.750400032PASS1. Both40correct/44jobs/11frames,
42actorsclosed/Manager0/allworkers-15/cgroupsabsent/zerothrottle. Primary
evaluation503cab9dd2a59024adc462c9ccd661e59098ded70513bc9aac389d01143761d1.
CPUdemand~1.54-1.64equivalents changedlittle; searchlatency growswithhistory.
TheseCPUfindings do not qualifyneuralmodels or erase210FAIL_STOP22/10/8.

212private tracingpreparationclosedafteractualcodec/GETPOST/portordinalproofs,
exactrestoration andprimary12pinreadback; no nativecapture. 215sourceaudit
foundretainedterminaljobs reobservedoneachservicetick, eachinvokingglobalpump;
causality remainsunproved.216ownsONEboundedpublicCPUcausalrepro, thenonly
provedminimalrepairwithordinaryfeatureTDD. Installedwatchersunchanged, voice
AssemblyAIactive/screenbatchdisabled. Full0167/0168 andhumancriteriaremainopen;
reservedfinalbroadreview/remediation0/0.


## Terminal-history repair | 2026-11-12

Integrated core8d4204aea (workerbe9b96df35670e365ff4d3d41c0e192f595d7e58):
maintenance skips reobserving immutable terminal requests while retaining the
initial manager pump and every nonterminal lifecycle path. Public real-worker
regression first failed at22 inspections, then passed;38 focused tests plus16
subtests pass. One matched fullCPU feedback: searchp95.212728581s and first
audio.240870962s versus earlier.751007146/.775913622. All40 rows correct,
11frame cursors preserved,42actors stopped,Manager0,workers-15/PIDs and cgroup
absent. Source handoffde24f72c77bf3a4af3d0308da76b87d7c4b079ab13cd9b833e21ebf973148038.
Primary full core regression is RUNNING, not yet passing evidence.

Private220/child221 prepare a separately scoped history22 successor guard;
they grant no native execution or freeze authority. Actual consumed native
starts remain22/10/8; old210FAIL/cohortSTOP is retained. Primary application
script stays11c3fc8c for historical validation; installed services unchanged.
Full0167/0168 acceptance and final broadreview/remediation0/0 remain open.


## Source gates validated | 2026-11-12

Primarycore402testsOK/49optional-live skips302.747s, actualexit0 at
testcommitd9a138487/production8d4204aea. Initial402-test failure is retained: an
old test expected terminal reobservation. Corrected fixture holds11frames active,
checks metadata-only observation, then verifies terminal output/cursor/replay
without further observation;21focused checks pass. ExistingResourceWarning
remains documented. Newguardchilddda47b83ff54f385af775c67326e3d0dba13acbd
passes85joinedappchecks16.65s against currentcore; old primaryapp11c3fc8c remains
available for historical evidence. Tickets219/221 close as source components.

Before native comparison,223 adds B2's actual document vectors and retained
owned-worker snapshots to its exported evidence after measurement.222 binds
independent verification to those artifacts;220 closes prospective source pins
against the resulting harness. No new native freeze or admission has occurred.
Actual native22/10/8, old210FAIL_STOP and finalreview/remediation0/0 persist.
Human word-timing scope is pending operator preference; full0167/0168 remain open.
