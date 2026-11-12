# Managed recording stage qualification (ticket0173)

`RecordingProcessor.advance` completes unfinished ASR units before starting
unfinished diarization units. This avoids switching GPU models for every unit.
Existing completed results, request IDs, unknown-outcome observation and explicit
retry lineage remain in the same durable checkpoint. One call still advances at
most one stage; preparation and publication behavior are unchanged.

This advances the vision's durable transcript/diarized-utterance and preserved
source-evidence outcomes. Maturity moves from source-tested alternating traversal
to source-tested model cohorts; native stage qualification is a separate gate.
A native output is not full mixed-load or useful-work acceptance.

## Source validation

From base `742e6fc`, public restart/cohort test first failed with observed profile
order `asr,diar,asr,diar`; after reversing the traversal nesting, all10 focused
processor tests passed, including lost acknowledgement, retry lineage, provenance
conflicts, ambiguity and publication custody.

```sh
flock /tmp/plan0095-tests.lock /absolute/transcribe-venv/bin/python \
 -m pytest -q tests/test_managed_recording.py
```

No hosted CI or final broad review is consumed by these checks.

## Native evidence boundary

Private prospectively frozen packets, manifests and logs live under the operator's
`~/.local/state/transcribe-audio/plan0095/native0173/` (0700), outside Git.
Native stages run through LocalAI's existing `register_recording_profiles`,
SharedService and owned default-JSONL ProcessAdapter with manager reservations
before model effects. A private exec wrapper preserves worker PID/parent custody,
sets four-CPU affinity and retains bounded-case diagnostics. Native cases serialize
on `/tmp/plan0095-native.lock` and run in a user systemd cgroup with12GiB RAM,
four CPUs,600seconds runtime and15seconds cleanup. Admission requires the declared
reservation plus3.5GiB free GPU memory. The total allowance is four physical stage
loads, with no unchanged failed repeat or unrelated service pause.

Retained smoke evidence as of this source checkpoint:

- Initial setup failed before physical loading because the systemd PATH lacked
  `nvidia-smi`; the launcher repair uses its absolute WSL executable path.
- The4GiB ASR profile was refused admission at7370MiB free; no physical model
  attempt occurred. Its configuration and refusal remain retained.
- Synthetic speech Pyannote Community-1 with1.5GiB reservation completed a real
  owned native stage: one turn and a256-dimensional speaker vector. The manager
  released all reservations and the worker PID was absent after cleanup.
- A distinct3.5GiB ASR candidate on a public59second AMI excerpt failed the loaded
  headroom floor during NeMo restore: minimum3580MiB free was below3584MiB.
  The failed attempt is retained; cleanup proved zero reservations and worker exit.

Synthetic speech smoke does not establish multi-speaker accuracy. The public AMI
cohort supplies human transcript words, speaker labels and segment boundaries;
its forced-aligned word timestamps are not human-reviewed word boundaries.
Native public-cohort results and any explicitly repaired ASR candidate are pending
at this source checkpoint. Full quality/mixed-load acceptance remains ticket0167;
this source change does not activate recording ingestion or publish transcripts.

The public59second DIA profile with upstream segmentation/embedding batch sizes32
failed its1.5GiB Torch allocator cap during speaker embedding (312MiB requested).
This is retained negative evidence, with confirmed release. A separately frozen
batchsize1 profile retains identical model files, dtype, cap and headroom floor;
its native result is recorded separately rather than replacing that failure.

The source Parakeet repair stages the checkpoint on CPU before CUDA transfer.
Installed NeMo otherwise moves the constructed model and the checkpoint state
onto CUDA together during restore. A focused model-boundary TDD test covers the
new load path; this source result does not qualify the repaired native ASR path.
No additional ASR native load is authorized within the four-load0173 allowance.

## Native closeout

Four physical model-load attempts were consumed; no additional native attempt is
permitted by this packet. The setup failure and4GiB admission refusal consumed
zero physical loads. All cases used the native lock and bounded owned cgroups.

| Physical load | Profile/input | Result | Process/reservation cleanup |
| --- | --- | --- | --- |
| 1 | Pyannote1.5GiB, batch32, retained synthetic3.435s speech | Native stage succeeded; one turn,256D vector | Proven |
| 2 | Parakeet3.5GiB, public AMI59s | Loaded headroom floor violated during restore; no ASR output | Proven |
| 3 | Pyannote1.5GiB, batch32, public AMI59s | Actual CUDA allocator OOM during embedding; no stage output | Proven |
| 4 | Pyannote1.5GiB, batch1, public AMI59s | External `nvidia-smi` read timed out after5s during loading; stopped without stage output | Proven |

The last case retained a minimum observed free GPU sample of8,204,058,624bytes
before telemetry failed; this does not establish uninterrupted capacity during
the missing observation. Its final fresh GPU read was8,716,812,288bytes free.
The owned worker PID was absent, all four recorded worker PIDs were absent, and
Manager reported no models and zero reserved bytes. The cgroup recorded
2,757,255,168bytes peak RAM and zero OOM events. Service runtime was85.748seconds.
No public-cohort speaker/word/timing quality metric is available because neither
public stage completed. No transcript, canonical output, routing or activation
was published.

The final batch1 packet SHA256 is
`36fe2f546d6bf4e7c7363672c8f18c153058564a400b0b75b9da46142f995d08`;
its manifest SHA256 is
`37394c5c114aab0fbf313b68e810ecc1c131127872f74339aed5c0a144d7de41`.
The private root retains every packet, original batch32 bundle/manifest,
separate batch1 bundle/manifest, physical-attempt ledger, per-worker diagnostic
logs, resource-control receipts, progress and final result JSON. Failed samples
remain failed; batch1 is an unqualified repair candidate.

The optional explicit harness is [managed_recording_native.py](../scripts/managed_recording_native.py).
It imports model/service code only on explicit execution and verifies cgroup
controls and frozen source hashes before admission. Run it only with a new
qualified packet and resource authority; ordinary source checks never invoke it.
The actual launcher shape used here was:

```sh
flock /tmp/plan0095-native.lock systemd-run --user --wait --collect --pipe \
 --unit=EXACT_FROZEN_CASE --property=MemoryMax=12G --property=CPUQuota=400% \
 --property=RuntimeMaxSec=600 --property=TimeoutStopSec=15 \
 /absolute/transcribe-venv/bin/python scripts/managed_recording_native.py \
 /absolute/private/packet.json
```

The harness currently uses the WSL executable `/usr/lib/wsl/lib/nvidia-smi`.
Its allocator cap and local GPU telemetry do not by themselves establish
Windows host capacity, global allocation enforcement or mixed-service acceptance.
The prospective packets pin interpreters, local model assets/packages, actual
worker source hashes, registered profile fingerprints, inputs and verdict bounds.

Remaining gates are repaired-Parakeet native execution under a separately
authorized allowance, reliable host GPU telemetry, completed public multi-speaker
DIA/ASR evidence and ticket0167 full managed useful-work acceptance. The processor
source optimization and CPUrestore source tests can integrate independently.
The source closeout uses local checks; hosted CI and the reserved final broad
review remain unconsumed. Integrated memory disposition belongs to the coordinator;
this worker did not write a separate Graphiti outcome.

## Explicit capacity source for successors

A new prospective packet may supply `telemetry` with exactly `endpoint`,
`host_id`, `gpu_uuid`, `token_path` and `expected_source`. The harness uses the
existing strict WindowsGpuTelemetry client, reads a private owned token in the
process, and retains source acquisition age using a monotonic clock. Selection
is explicit: there is no automatic source fallback or cached-good capacity.
Legacy packets without this field retain their direct NVIDIA acquisition path.
Both isolated stages and full recording traversal enforce the loaded headroom
floor. This source change alone does not qualify DXCore or any native model.

Public capacity-selection test red (missing provider argument), then11focused
source tests pass; invalid/private-token and unavailable samples fail closed.
No successor native effect occurred as part of this harness adaptation.

## Independent DXCore successor

The exhausted four-load packet remains unchanged. A separately authorized
successor permits two additional physical loads, cumulative limit6, after the
independent Windows DXCore sampler qualification. This changes acquisition and
ASR restore custody; it does not replay or erase the failed NVIDIA-reader cases.

The harness accepts `max_physical_stage_load_attempts: 6` only with
`limits.case_seconds` at most160 and cleanup at most15seconds. The deadline covers
the entire case, including setup and all stage transitions; it is not reset when
a model changes. Legacy packets retain their original four-load/600second bounds.
Each successor contains one native stage. The independent composite controller
owns a fresh bounded Windows guardian/tunnel and WSL unit for each case, and its
remaining-lifetime gate must pass before native argv is released. No180/186second
Windows guardian fuse is increased.

Successor packets select `telemetry_config_path` pointing to the composite
controller's prepared private `telemetry.json`. The configuration carries only
the loopback endpoint, pinned host/GPU/source and private bearer-file path; bearer
contents never appear in argv or packet JSON. It is exclusive with inline
`telemetry`, must be a private owned regular singly linked file, and is included
in the frozen source/input hash map. Its source is explicitly `windows-dxcore`.
The capacity client retains validated acquisition age, generation, sequence and
source time; failures never fall back to Linux NVIDIA telemetry. Load and inference
continue to enforce the3.5GiB headroom floor against fresh strict samples.

Public source tests first failed on the missing successor-limit seam, then on a
161second successor wrongly accepted, on lost source age, and on missing private
configuration-file selection. The resulting focused lane passes13tests. These
source results authorize no model effects before the composite controller is
source-validated and the new prospective native packets are frozen.

## Freshness-contract reconciliation

The first independent-DXCore successor retained a stricter2second candidate cutoff
introduced by the native harness. The accepted sampler recipe and Manager contract
allow a maximum validated acquisition age of5seconds. Public-cohort DIA reached
real inference but was stopped when the extra2second constraint failed; it remains
a failed sample. Its rejected raw age was unavailable from the public client API,
so this record does not claim that the sample would have passed a5second limit.
The unused4GiB ASR case was refused on fresh byte-based capacity, and its distinct
3.5GiB CPU-staged replacement stopped before model loading on the controller's
TIME_WAIT transport defect. All failed/setup-only roots and cleanup receipts remain.

A newly authorized contract-corrected pair permits cumulative physical loads7
(prior5 retained, two new stages). Packets explicitly set
`telemetry_max_age_seconds: 5`, validated finite and in `(0,5]`. Legacy packets
continue to default to2seconds. No rejected sample supplies capacity, no Linux
fallback is introduced, and no old failure is reclassified. This restores the
already qualified source contract for the new cases; it does not increase the
accepted age beyond5seconds. Native140second whole-case/15second cleanup bounds,
model caps,4CPU/12GiB resources and the3.5GiB headroom floor stay explicit.

Future strict-client failures retain a sanitized rejection descriptor (error type,
expected source, configured age limit and `rejected_sample_available: false`) plus
last accepted metadata. The public telemetry API does not expose rejected raw
responses, so the harness does not intercept private provider methods or log
bearer data to fabricate that evidence. Source tests first failed on the new load
bound, missing configurable age and missing rejection descriptor, then all15
focused processor/harness checks passed. New model effects remain gated on the
corrected source-qualified composite controller and fresh prospective packets.

Earlier source qualification was separate from hardware admission: Native ASR
produced a live canary transcript but capacity publication exceeded its original
five-second contract. That result led to LocalAI ticket0179 and the later ticket0180 source repair.
Existing voice/Sound routes remain Assembly-only; the screen queue remains paused.
## Cached-factory under-load DIA canary

A new operator-authorized case used the source-qualified ticket0180 cached-factory
recipe, rather than the superseded per-acquisition factory implementation. Idle
qualification was treated as a prerequisite, with under-load behavior measured
separately. The stable app harness matched source `1590585` byte-for-byte; no model,
existing service, precision, floor or accepted resource limit changed.

The prospective native packet SHA256 was
`fcbe24bd3d507b3cc37d3836ee5d12a4c0a0f1bd550f31ecf6ff9bbe1bc795de`;
composite controller plan SHA256 was
`81d793c96cf9717f2925506fd94e188e30e80645dd0463e27cb2bc85b430bdcf`.
The case retained the1.5GiB Pyannote cap, both batch sizes1,4CPU/12GiB,
140second native case plus15second cleanup,5second strict acquisition bound and
3.5GiB free-memory floor. It consumed physical load6 of cumulative maximum7;
all five earlier physical outcomes and setup-only failures remain retained.

The real public59second AMI job succeeded after85.591seconds, returning12turns
and three anonymous local labels with full finite256-dimensional vectors. Its
profile identity was
`7f4f54348dd7278fb1d8046f44895e8eae05729e00cfad45c532b7613b35f5e1`.
All72 retained strict capacity observations stayed within the source/floor gates:
maximum effective acquisition age0.456072seconds; minimum free5,418,995,712bytes.
A pinned private wrapper captured97 additional nonce-bound diagnostic HTTP rows,
inside the same owned unit. Every row was explicitly non-admission evidence.
All97 sequences were unique within one generation, sampled iterations16..506
with observed phase `sleep_enter`. Maximum raw sample age was218.952ms and query
duration2.0481ms. The diagnostic thread stopped and joined. These observations
support live publication during this case; HTTP progress never refreshed capacity
age or substituted for a rejected strict sample.

Actual DIA quality was measured against human AMI transcriber segment evidence:
overlap-inclusive zero-collar DER was0.1530782443, comprising6.018625seconds missed
and1.879906seconds confusion over51.598reference speaker-seconds. Three model
labels mapped to three of four reference speakers. Complete timestamp MAE was
unavailable because two reference activity boundaries had no mapped speaker.
ASR was not run and no ASR WER is claimed. The retained raw scorer's empty-word
placeholder is not an observed ASR result. These are measured DIA outcomes, not
full mixed-load or useful-work acceptance.

**The overall native canary failed bounded cleanup.** After15seconds the Manager
still retained the1.5GiB model in `stopping`; adapter close also exceeded its
second2second wait after SIGKILL. The independent unit subsequently proved its cgroup empty. Windows
source teardown reported an empty job and absent guardian, but the exact held
payload handle was not signalled within its wait (`wait_result=258`), so its
original cleanup receipt remained unproved. A later bounded read-only check found
that exact payload PID and guardian absent; it did not recover a held numeric exit
or retroactively pass the original release budget. No uncertain owner was signalled
and no Manager reservation record was rewritten by this agent.

Private evidence is under `native0173/factory180-dia2-controller/` and adjacent
`factory180-dia2-*` artifacts. Full stage output/vectors remain in
`packet-factory180-dia2-result.json`; quality, publication and adjudication JSON
keep output, acquisition, quality, bounded release and later absence separate.
Further native dispatch remains held for primary cleanup reconciliation. ASR also
requires fresh capacity at its declared profile plus the unchanged floor.

Current gate: LocalAI ticket0181 implements early owned-worker unload escalation.
A newly pinned isolated DIA canary must prove bounded release before subsequent
GPU qualification. Production Assembly-only routes and paused screen queue persist.
## Prepared release-escalation successor

Ticket0181 adds inspection-driven SIGKILL after the existing two-second TERM grace;
its source/OS checks do not prove the historical GPU exit delay is fixed. A separately
authorized successor permits cumulative physical loads8 (six outcomes retained,
two remaining stage proofs), without resetting the ledger. The harness retains
legacy7/default4 behavior and rejects counts above8; native140second/15second,
4CPU/12GiB,5second source-age and3.5GiB floor bounds remain unchanged. The strict8
source seam first rejected8, then all16 focused app checks passed. A fresh cached-
factory DIA release canary must be prospectively frozen and admitted by the primary;
source validation and prepared metadata are not model-dispatch authority.

## Admitted ticket0181 release canary

The new stopping-inspection escalation passed source/OS tests and was integrated
before a separately admitted DIA canary. Original packets and the six preceding
physical outcomes were retained. The new native packet SHA256 was
`a691673ac947900bb3a9c6c8815b3cc0746f9fc5f17e25a2664451b960ef15ff`;
controller plan SHA256 was
`dd0429cddf89dd0597f7883c5c72a0e9d3dc70697ec1e3145c1399888695b2fa`.
It used integrated app `bd28f5d` and LocalAI adapter code `5105d9c19`, with the
cached0180 recipe, unchanged1.5GiB/batch1 model profile,5second freshness and3.5GiB
floor,4CPU/12GiB,140second native case and15second app cleanup. This was physical
load7 of cumulative maximum8; no next model or retry followed its failure.

Real DIA completed after59.578seconds with12turns and three finite256D speaker
vectors. Reference scoring again measured DER0.1530782443; complete timestamp MAE
remained unavailable because two reference boundaries lacked a mapped speaker.
ASR was not run. All53 retained strict capacity samples stayed fresh and above the
floor (maximum effective age0.369067seconds, minimum free6,380,339,200bytes).
The separate diagnostic-only lane retained75 nonce-bound HTTP200 rows with75unique
sequences, all observed phase `sleep_enter`; it did not supply capacity decisions.

**Bounded app release failed again.** After15seconds, Manager still retained the
1.5GiB `stopping` model and the result's exit readback recorded its leader present.
The known custody contained only that leader/group and no recorded descendants.
The later explicit close reached its second two-second wait after requesting
SIGKILL and timed out. No shutdown-state/kernel-wait snapshot or numeric model
exit was captured, so this does not prove a driver cause, ignored signal, or a
leader-reaped/child-alive scenario. The earlier actual OS observation was a running
leader during loading; it is not shutdown evidence.

Independent owned-unit cleanup subsequently proved an empty Linux cgroup and the
leader was absent on current readback. Windows source teardown succeeded on this
case: the exact held payload handle signalled with numeric exit1, its job was empty,
guardian/payload were absent and the owned tunnel was absent. This later cleanup
does not retroactively pass the failed app15second release budget. No historical
receipt or Manager journal was rewritten.

The compact private adjudication is `native0173/release181-dia-v2-adjudication.json`
(SHA256 `344be6e7417dfd593d827605a4ce4ad6245fe7b195aef4145257af29dbcd8870`).
Raw stage output/full vectors, strict observations, diagnostic rows, custody and
independent cleanup remain adjacent to `release181-dia-v2-controller/`. The overall
verdict is native-output success with bounded-release failure. Further dispatch is
held; ticket0182 is gathering shutdown-boundary diagnostics without changing signal
or deadline policy. Source0181 validation is not a verified native behavior cure.


## Prospective diagnostic successor after physical load7

The seventh attempt retained real DIA output but failed bounded app release. A
shutdown diagnostic source change is tracked in LocalAI0182; it changes no signal
policy or exit proof. Before any next effect, the successor allowance is reframed
as cumulative9=seven retained physical outcomes+at most two new stage proofs. One
new DIA diagnostic/release case must run first and stop on its first failure;
only a passing release could unlock the remaining ASR proof. No source change
alone admits either run. All prior frozen packets and attempt history remain.

The public native_limits seam rejected9 before the one-line limit update; the
focused processor/harness16tests passed in1.75s afterward. Counts above9 are
rejected, legacy4/7/8 packets remain valid, and the successor runtime<=160s,
cleanup<=15s,4CPU/12GiB, strict5s age and3.5GiB floor remain unchanged. These normal
feature tests do not consume the reserved final review/remediation counters.
## Ticket0182 shutdown-boundary result

The separately admitted diagnostic case used LocalAI `89e57c02e`, app `b12518e`,
native packet SHA256 `7fa33c796f0de4eee949714bd28372bb20e7a456347147cf57118c4d498822c1`
and controller plan SHA256
`6f806175fa53ad073085d100bf262b083ae0bc30f4b0d1ecdd793ac97102a0dc`.
No signal, deadline, model math or resource limit changed. This consumed physical
load8 of cumulative maximum9 and stopped at its first failure; no next model ran.

Real output again contained12turns and3x256D vectors, with measured DIA DER0.1530782443.
Timestamp MAE remained unavailable for two missing boundaries; ASR was not run.
All72 retained strict source samples passed (maximum effective age0.307245seconds,
minimum free5,611,892,736bytes). The diagnostic-only HTTP capture retained95nonce-
bound unique publications, with94`sleep_enter` and one`list_enter` observation.

The new generation-bound records capture the release boundary: before TERM the
leader was sleeping in `anon_pipe_read`; its first wait snapshot then showed
leader state`Z`, shared pending mask`0x4000`, no blocked signals and `returncode=None`.
A thread in that same thread group remained state`D` in `dxgvmb_send_sync_msg`.
SIGKILL requests returned at approximately2.043,4.056 and17.110seconds after stop
began; subsequent waits still captured that thread and no numeric reap. Custody
recorded no descendant processes. The optional PGID/SID snapshot matched the leader
while loading; no shutdown PGID snapshot was available. Underlying driver cause
remains unproved.

`stop_owned` recorded a4.028second closer timeout,988pumps and15.002seconds total,
retaining the1.5GiB stopping reservation. The overall bounded-release verdict failed.
The independent Linux unit subsequently proved an empty cgroup. Windows original
held-handle cleanup was unproved; a later exact read-only check found its payload
and guardian absent without reconstructing an exit signal/code. No original failure
or Manager state was rewritten, and no extra OS signal, GPU control, service change
or model/collector start followed the failure.

Private `native0173/diag182-dia-adjudication.json` SHA256
`ca8de6974892f76ab61a7dcf9e3772464d2809b8afa3bd2650a180e5c92a040d`
links the retirement-generation file, stop timing, raw stage/vectors, quality,
publication and original/late cleanup records. Further effects remain held pending
new primary authority and a prospectively frozen packet. Memory disposition:
`forbidden` by the delegated task's explicit no-memory-write boundary.

The operator-approved maintenance0182 control consumed physical attempt9. With
Speech temporarily paused, real DIA again succeeded (12 turns and three speaker
vectors), but the same 15-second application release gate failed. The retained
retirement receipt shows the matched leader949024 in Z, same-process thread952279
in D at `dxgvmb_send_sync_msg`, and no numeric return code after returned SIGKILL
requests and timed waits. `stop_owned` recorded a4.055second closer timeout,
978pumps and15.007seconds total; the1.5GiB stopping reservation remained. This
reproduces the observed release failure under the maintenance condition; it does
not establish a driver cause or a verified remedy. Independent outer Linux and
Windows cleanup both proved true. The coordinator's restoration receipts record
Speech and both model workers ready before retiring the backup restore timer.

Private evidence remains under `native0173/maintenance0182-dia-controller`,
`packet-maintenance0182-dia-result.json`,
`workers-maintenance0182-dia/diarization/retirement-33a476ff826954e2.json`, and
`packet-maintenance0182-dia/manager/stop-owned-diagnostics.json`; maintenance and
restoration receipts remain in `host-release-diagnosis`. Previous failed-release
receipts and all nine physical starts remain immutable.

The harness now accepts an explicit cumulative maximum11, preserving prior9 plus
at most two prospective DIA/ASR proofs. Legacy4/7/8/9 packets remain valid;12 is
rejected. The single whole-case successor ceiling160seconds and cleanup ceiling
15seconds are unchanged. This source change authorizes no execution: a successor
must bind the new183 worker source and receive exact prospective admission. No new
native packet was frozen or executed in this unit. Memory disposition: `forbidden`
by the delegated no-memory-write boundary.

Graceful183 DIA v2 consumed physical10 with production Speech running. Real DIA
succeeded with12turns and3vectors; strict DXCore54samples had maximum age0.265383s
and minimum free6266793984bytes. Application release still failed: no numeric143
exit was observed, the initial2s graceful wait expired, and SIGKILL fallback returned.
The matched leader1069267 becameZ while same-process thread1069979 remainedD in
`dxgvmb_send_sync_msg`, with pendingKILL and returncodeNone. `stop_owned` retained
1.5GiB after15.002s (closer4.023s TimeoutExpired,953pumps). Normal-close source
capability therefore did not establish native release. Driver cause remains unproved.

The outer Linux cgroup cleanup proved true. Original Windows cleanup remained false
(wait258, held payload not signalled). One later read-only reconciliation found exact
payload49620/FILETIME134389023685060165 and guardian absent, and Linux
leader1069267/start9620833 absent. It did not reconstruct numeric exit or revise the
failed15s release. Private original evidence remains under
`native0173/graceful183-dia-v2-controller`, including late-readonly Windows/Linux
receipts, plus `workers-graceful183-dia-v2/diarization/retirement-43fef582b36d37d8.json`
and `packet-graceful183-dia-v2/manager/stop-owned-diagnostics.json`.

The next explicit source allowance is cumulative12: ten immutable physical starts
plus at most two prospective DIA/ASR proofs. Public tests accept12 and reject13,
preserving legacy4/7/8/9/11 and the unchanged160s successor/15s cleanup ceilings.
No new native packet was prepared pending source184. No GPU, collector, service or
signal effects occurred in this follow-up; memory disposition remains `forbidden`.

EOF184 consumed physical11/12 with production Speech running. Real DIA succeeded
(12turns,3speaker vectors), but application release failed again:15.005seconds,
998pumps, closer4.023seconds TimeoutExpired,1.5GiB stopping reservation retained.
After the2second wait, fallback SIGKILL returned; matched leader1328272 becameZ
while same-process thread1329365 remainedD at `dxgvmb_send_sync_msg`, with pending
KILL and no numeric exit. No143 or exit-observed proof was retained. The rolling
16record diagnostic bound omitted early TERM/EOF records, so the pinned
`eof_on_stop=True` policy cannot establish observed EOF delivery from that receipt.
This is output-positive/release-failed, not a verified lifecycle remedy.

Independent outer Linux and Windows cleanup both proved true. The coordinator's
later exact read-only receipt found leader1328272, blockedTID1329365 and the original
cgroup absent; this does not revise the failed15second Manager release. Strict
53samples had maximum age0.306030seconds and minimum free7415619584bytes. All
lifecycle variants remain held at the host gate; no further model, collector,
service, reset or restart effect is authorized by this result.

Private `native0173/eof184-dia-adjudication.json` SHA256
`a96e538be6c75ccd0d1f519b5278aecc6099bbb3ad00fe59750350cecada6b44`
links the frozen packet, real output, controller cleanup, generation retirement,
stop timing and `eof184-dia-controller/independent-thread-cgroup-readback.json`.
All11physical starts and prior failures remain immutable. This closeout made no
native preparation or runtime change. Memory disposition: `forbidden` by task.

Ticket0221 adds a source-only prospective V4 recording guard for exactly22
retained physical starts and at most3 further fixed cohort cases (M1, B2, M3),
with stop on first failure. It advances bounded, reproducible qualification;
source preparation does not change native acceptance maturity or admit execution.
The explicit max25 packet requires `budget_contract` equal to
`independent-clients-v4-role-data-v2-history22`, layout
`INDEPENDENT_CLIENT_RECEIVERS_V4`, observation protocol
`ROLE_DATA_ENDPOINTS_V1`, integer `retained_physical_starts: 22`, and
`allowed_stages: [diarization]`. Its `retained_history_sha256` comes from
`foreground_cohort_history22_sha256`: canonical compact sorted JSON of the first22
ledger entries, rejecting short/non-list/nonfinite history. Raw file hashes and
first21 digests do not satisfy this contract. Future appends preserve the prefix;
changes to entry22 invalidate it before capacity acquisition or worker execution.

Legacy16/19/22/24 guard branches and the first21 helper remain unchanged. Max25
requires the new scope; max26 is rejected. Successor runtime remains at most160s
and cleanup at most15s. The companion220 custody contract owns the exact fixed
case order and embedding/TTS ceilings13/11; this recording guard binds recording
history and ceiling25 only. TEMP public tests reach a sentinel capacity boundary,
never a worker, while negative tests refuse before that boundary. A read-only
actualfirst22 fixture can be supplied using `TRANSCRIBE_V4_HISTORY22_FIXTURE`;
only its TEMP copy is passed to the guard. Historical ledgers, keys, installed
services and native allowances are not modified. Memory disposition: `forbidden`.
