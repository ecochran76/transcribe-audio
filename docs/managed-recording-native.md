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
