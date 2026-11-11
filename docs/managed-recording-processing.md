# Resumable recording processor

`managed_recording.RecordingProcessor(config, client=managed_client)` is the
CPU-only application boundary for ticket0164 / Plan0095. It does not import models,
start workers, schedule GPU capacity, convert media, or publish canonical files.
It advances vision outcome 1 (durable transcript and diarized utterances) from an
absent composition to isolated implementation; native shadow/pilot proof remains
outside this module.

Configuration requires `state_directory` and `spool_root` (private directories),
`unit_seconds` (positive, at most 3600), `profiles` containing `asr` and `diarization`
objects with `profile` and 64-character SHA256 `identity`, and prospectively reviewed
`speaker_similarity` (cosine, -1 to 1) and positive `speaker_margin` (up to 2).
The caller must choose unit bounds from qualified profile limits and yield costs;
the duration ceiling here is only a safety bound. No threshold is quality-qualified
by the implementation tests.

```python
processor = RecordingProcessor(config, client=client)
key = processor.prepare(
    pcm_wav_path, recording_id=recording_id, conversation_id=conversation_id,
    metadata=source_and_calendar_metadata, options=stage_options,
    prior_artifacts=previous_artifact_references,
)
checkpoint = processor.advance(key)
# Persist key in the caller's recording queue. Repeat advance after pending waits.
# A ready checkpoint permits guarded publication:
receipt = processor.publish(key, publish_generation)
```

`prepare` accepts only PCM16 mono16k WAV. It streams bounded frames into immutable
content-addressed WAV files and persists every stage intent before any submission.
The generation includes source bytes/path, original recording/conversation IDs,
metadata, prior artifacts, profile fingerprints, options, unit offsets and controls.
A changed identity creates a distinct generation. `snapshot(key)` exposes retained
raw stage results and states. Private checkpoints are atomically replaced and
fsynced; per-generation locks exclude concurrent effects.

`advance(key)` performs at most one stage submit/observe/result operation and returns
a detached checkpoint. After all stages finish, one additional call reconciles the
recording and sets `state=ready`. Callers own bounded polling/backoff and scheduling.
Requests use historical workload class and the frozen basename/hash/options payload.
The injected client implements `submit(request)`, `observe(request_id)` returning
`{state: ...}`, and `result(request_id)` returning `{state: succeeded, result: ...}`.
Results follow the frozen shared stage contract and include
`model: {identity: <pinned profile fingerprint>}`. Timestamp bounds, stage duration,
finite nonzero evidence and pinned model identity are checked before checkpointing.

Lost submission acknowledgements remain observe-only across restart. The sole
implicit replay exception is explicit `queue_full`/HTTP 429 rejection. Terminal
failed/cancelled stages require `retry_stage(key, unit_index, stage)`; that method
observes confirmed terminal custody before creating a new stable attempt ID and
retains its predecessor. Completed stages are never retried. No unknown result
or publication is automatically replayed, cancelled, or redirected to another route.

Recording output contains original identity fields plus:

- `words`: original ASR word fields, recording-relative `start_seconds` and
  `end_seconds`, and `unit` index.
- `turns`: original diarization turn fields, recording-relative seconds, `unit`,
  retained `local_speaker`, and recording-level anonymous `speaker` or null.
- `unresolved`: unit/local-speaker, explicit missing/ambiguous/competing-evidence
  reason and candidate cosine evidence.

Vectors must come from the same pinned diarization model. Numeric label equality
never creates continuity. Different vector dimensions, missing evidence and close
competitors remain unresolved. Distinct labels within one unit cannot silently
collapse into the same anonymous speaker. The first qualified vector anchors each
anonymous speaker; this conservative deterministic policy is not named-person
identification, clustering quality qualification, or overlap-window deduplication.
Units partition the source exactly without overlap, so offsets preserve the source
frame timeline without generating duplicate overlapping windows. Word-to-turn
utterance construction and canonical DOCX/TXT/JSON/library writing belong to the
consumer integration.

`publish(key, callback)` passes a detached complete output to the callback, which
must preserve previous versions and atomically commit the canonical generation
and source/conversation metadata. It must return a nonempty JSON-serializable
receipt. The processor persists publication intent before calling it; subsequent
successful calls return the retained receipt without calling the writer again.
Exceptions, missing receipts or crashes leave an uncertain marker and prevent
replay. External reconciliation of uncertain canonical effects is deliberately
required; this API does not pretend filesystem and external store writes share a
transaction. Never overwrite an old artifact inside the callback merely because
it is listed in `prior_artifacts`.

Local synthetic public-boundary tests cover lost acknowledgements/restart,
checkpoint retention, global offsets, same-model continuity, explicit ambiguity,
profile conflict, terminal retry lineage and uncertain publication. These tests
are source qualification only. No real media CLI, private recording, native model,
GPU custody, quality or installed-runtime acceptance is claimed. Hosted CI is
skipped under operator policy; the coordinator owns joined integration and native
smokes, including canonical artifact reopening and prior-version retention.

Implementation validation receipt (base source `ac65dad`):

- Red: root venv `python -m pytest -q tests/test_managed_recording.py` initially
  failed collection because the new public processor module did not exist.
- Subsequent public behavior reds caught mismatched vector dimensions incorrectly
  becoming a new speaker, competing same-unit labels silently collapsing, and a
  missing stage field escaping as KeyError. Each was corrected before proceeding.
- Green: `flock /tmp/plan0095-tests.lock
  /home/ecochran76/workspace.local/transcribe-audio/.venv/bin/python -m pytest -q
  tests/test_managed_recording.py`: **9 passed**.
- Regression green: `PYTHONPATH=/tmp/localai-plan0095-integration/qualification
  flock /tmp/plan0095-tests.lock
  /home/ecochran76/workspace.local/transcribe-audio/.venv/bin/python -m pytest -q
  tests/test_managed_recording.py tests/test_managed_transcription.py`:
  **14 passed**. An initial run without LocalAI on PYTHONPATH failed its existing
  dependency import; this environment setup issue is resolved by the command above.
- `git diff --check`: passed. Linux CPU synthetic WAVs and an injected client replace
  hosted CI and external service/model operations. CLI/manual native export smoke
  remains with the separately authorized joined integration packet.
