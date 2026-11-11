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
