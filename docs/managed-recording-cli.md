# Managed recording CLI

`managed_recording_cli.py` is an explicit application entry point for ticket0169,
Spec0005 and Plan0095. It prepares media with CPU ffmpeg, calls the authenticated
LocalAI `ManagedClient` through `RecordingProcessor`, reconciles the recording,
and publishes versioned DOCX/TXT/JSON. It does not load a GPU model, choose a
fallback backend, rename source media, or activate a historical queue. This builds
the vision's durable diarized transcript path at isolated implementation maturity;
representative native quality, custody and pilot acceptance remain separate.

```sh
python managed_recording_cli.py /private/original-recording.m4a \
  --config /private/managed-recording.json \
  --recording-id EXISTING_RECORDING_ID \
  --conversation-id EXISTING_CONVERSATION_ID \
  --existing-transcript /private/previous-transcript.json
```

`--existing-transcript` is optional. When supplied, its immutable JSON snapshot
provides calendar/event metadata, recording times, title, prior-artifact references
and IDs; explicitly supplied IDs must agree. Without it, both IDs must be explicit.
No calendar lookup, identity inference, source mutation or old-artifact rewrite
occurs. Source media path/name and SHA256 remain separate from the decoded upload
representation. The original is rehashed after decoding and before publication.

Use private owned configuration and token files (mode 0600), and private owned
state/spool/output directories (mode 0700). Configure registered stage fingerprints,
reviewed unit bounds and speaker thresholds explicitly:

```json
{
  "endpoint": "https://manager.example.invalid",
  "token_file": "/private/consumer.token",
  "ca_file": "/private/manager-ca.pem",
  "output_directory": "/private/transcript-generations",
  "request_timeout_seconds": 10,
  "decode_timeout_seconds": 300,
  "wait_seconds": 300,
  "poll_seconds": 0.25,
  "max_stage_attempts": 1,
  "options": {},
  "processor": {
    "state_directory": "/private/recording-state",
    "spool_root": "/private/authorized-manager-inputs",
    "unit_seconds": 30,
    "speaker_similarity": 0.9,
    "speaker_margin": 0.1,
    "profiles": {
      "asr": {"profile": "registered-asr", "identity": "<64 lowercase hex characters>"},
      "diarization": {"profile": "registered-diarization", "identity": "<64 lowercase hex characters>"}
    }
  }
}
```

Values above illustrate syntax, not qualified native thresholds or unit sizes.
The actual profile's supported limits and reviewed evidence packet govern them.
`ca_file` is optional for normal system trust. TLS verification cannot be disabled;
only disposable literal-loopback HTTP tests may set `allow_insecure_loopback=true`.
`ffmpeg` must be on PATH and the installed LocalAI qualification package must be
importable. There is no environment installation or automatic model acquisition.

The importable public boundary is
`run(source, config, *, recording_id=None, conversation_id=None,
existing_transcript=None) -> receipt`. `main(argv)` returns exit 0 with one JSON
receipt on stdout; failures return exit 1 with a diagnostic on stderr. Credentials
are read privately and never passed to subprocesses or included in receipts.
`attribute_words(words, turns)` is the public deterministic attribution boundary.

CPU decoding produces a separate PCM16 mono16k WAV under the private CLI state
namespace. The processor creates immutable bounded content-addressed unit WAVs.
An immutable invocation manifest binds original source path/hash, recording and
conversation IDs, configuration, output/store paths and the old transcript snapshot
to the processor generation. Changing configuration or metadata for that invocation
fails before new managed work; keep its state directory to preserve this guard.
Changing the original source bytes creates a new retained generation. Moving the
state directory is an explicit new custody namespace, not checkpoint migration.
An uncheckpointed decoded artifact after a crash is a reconciliation gate rather
than an excuse to overwrite it.

Polling and request/decode timeouts are finite. The outer wait is bounded to at
most 86400 seconds, polls to 0.01..60 seconds, decode to 1..3600 seconds and attempts
to 1..10. An in-flight HTTP call can consume its own configured timeout after the
outer wait boundary; no service job is cancelled just because the caller exits.
Known queue rejection may retry the same ID. Lost acknowledgements are observe-only,
including after restart. Confirmed failed/cancelled custody permits a new stable
attempt only within `max_stage_attempts`; previous attempts remain in the checkpoint.
No uncertainty triggers a GPU route fallback or implicit replay.

A word receives an anonymous speaker only when its entire span is covered without
gaps by turns of exactly one resolved speaker. Missing evidence, competing speakers,
overlap and uncovered spans are `UNKNOWN`; proximity or midpoint alone cannot
assign them. Output utterance timestamps and nested word timestamps are milliseconds,
matching existing transcript artifacts. Raw unit-relative evidence and reconciled
recording-relative second timestamps remain retained in the managed checkpoint
and the artifact's `managed_recording` field.

Publication creates `output_directory/recording-<processor-generation>/` containing
`transcript.docx`, `transcript.txt`, and `transcript.json`. Every generation has
its own directory; earlier generations and supplied prior artifacts remain intact.
The JSON uses existing schema 2 recording/conversation/source/event/utterance fields.
DOCX/TXT use the same utterance sequence. Receipt includes IDs, generation, output
paths and SHA256 hashes. Original metadata is retained in the managed evidence;
calendar metadata is also visible in DOCX and the canonical JSON event field.

The processor persists publication intent before any writer effects. After success,
repeat execution returns the same retained receipt without duplicate writes or
store commits. A failure after any export or store side effect remains uncertain
and blocks replay. Do not delete that marker to retry; reconcile actual artifacts
and store effects through a separately reviewed repair. Cross-filesystem/store
publication is not falsely advertised as one transaction.

Optional existing-store ingestion requires explicit configuration:

```json
{"transcript_store": {"root": "/private/transcript-library", "embedding_provider": "debug-hash"}}
```

This bounded entry point currently permits only CPU `hash`/`debug-hash` embeddings,
never implicit Ollama/direct GPU embedding effects. It calls the existing
`transcript_store.ingest_artifact` API after durable exports and returns the real
store receipt. Debug embeddings support deterministic store plumbing, not native
retrieval quality acceptance. Production embedding-provider composition requires
its own managed boundary.

Validation uses actual CLI subprocesses, CPU ffmpeg, authenticated disposable
loopback SharedService HTTP, and manager-owned scripted JSONL CPU engines. These
checks qualify the application and protocol composition, not native speech models,
GPU release, private media, installed services or output quality. Hosted CI is
skipped under operator policy. The joined native packet must reopen exports,
review words/speakers/times and verify actual library/provenance and resource release.

Implementation receipt (base source `b6caeb83013daa43d35cde886e450a6b79c1fee0`):

- Public red: after correcting the external fixture's required operator argument,
  the first actual CLI subprocess test failed because `managed_recording_cli.py`
  did not exist. Implementing that entry point made the same public scenario green.
- Focused green: **8 passed in 11.70s** using root venv, LocalAI qualification on
  PYTHONPATH and `flock /tmp/plan0095-tests.lock`.
- Combined green: `PYTHONPATH=/tmp/localai-plan0095-integration/qualification
  flock /tmp/plan0095-tests.lock
  /home/ecochran76/workspace.local/transcribe-audio/.venv/bin/python -m pytest -q
  tests/test_managed_recording_cli.py tests/test_managed_recording.py
  tests/test_managed_transcription.py`: **22 passed in 39.07s**.
- CLI subprocess environments prepend `/home/linuxbrew/.linuxbrew/bin` for real
  ffmpeg. Test media are disposable generated PCM; scripted workers are CPU-only.
  The smoke reopens DOCX/TXT/JSON and the optional real store artifact, verifies
  source/prior artifact bytes, receipt identity, configuration conflict, terminal
  retry lineage, word gap/overlap handling and uncertain side-effect suppression.
- `git diff --check`: passed. No hosted CI, model/GPU/private-media operation,
  installed service mutation or native acceptance occurred.

## Resumable stage-result budget

Use `--max-stage-results N` (integer1..10000), or the public
`run(..., max_stage_results=N)` keyword, to return after at mostN newly durable
unit-stage results in that invocation. Omit it for the existing completion path.
For a large recording inside an independently bounded case:

```sh
python managed_recording_cli.py /private/full-recording.wav \
  --config /private/managed-recording.json \
  --recording-id EXISTING_RECORDING_ID \
  --conversation-id EXISTING_CONVERSATION_ID --max-stage-results 1
```

A budget return is successful resumable progress (exit0), with a receipt shaped as:

```json
{"state":"partial","published":false,"resumable":true,
 "reason":"max_stage_results","processor_key":"EXISTING_KEY",
 "checkpoint_path":"/private/state/EXISTING_KEY.json",
 "cli_manifest_path":"/private/state/cli/CLI_KEY.json",
 "recording_id":"EXISTING_RECORDING_ID","conversation_id":"EXISTING_CONVERSATION_ID",
 "new_stage_results":1,
 "completed_stage_results":[{"unit_index":0,"stage":"asr","request_id":"EXISTING_REQUEST_ID","attempt":0}]}
```

Resume using exactly the same source, configuration, metadata and IDs. Completed
results and request IDs are reused; the new budget counts only results completed
since this invocation began. The execution budget is a flag/keyword, not an
immutable configuration field, and does not change source/profile/checkpoint
identity. All source units remain in the checkpoint. Partial receipts contain no
published output paths and never authorize publication or a stage retry.
When the final stage result reaches the budget, offline reconciliation and normal
DOCX/TXT/JSON publication finish in that invocation; an extra empty run is not
required. Existing case deadlines and external resource/admission controls remain
in force. An interrupted or failed operation retains its ordinary recovery rules.

Explicit `queue_full` (HTTP429) and `retention_full` (HTTP503) rejections leave
the exact chunk request pending, with its original ID and attempt. They prove
that admission did not occur. Other transport errors remain unknown and are
observed without resubmission. This allows a paused chunk stream to resume when
manager admission becomes available without replaying completed work.

### Explicit repaired-profile retry

`RecordingProcessor.retry_stage(key, unit_index, stage,
replacement_profile={"profile": "registered-repair", "identity": "<64-hex fingerprint>"})`
creates a new attempt only after observing the old request as failed/cancelled.
The replacement must be explicit and pinned. Original request/profile evidence
stays in attempt history; completed results and initial frozen configuration
are unchanged. Results are validated against the new attempt's fingerprint.
This does not permit replay of an unknown outcome, automatic retry, or changing
an existing registered profile. Consumers must retain the attempt lineage when
reporting mixed-profile recovery.

Source hashing retries an `ENOMEM` filesystem read at the same byte offset,
with at most three attempts per block. It preserves the full source SHA-256
check and stops on persistent errors or other error codes. This is a bounded
read-only transport recovery, not a GPU/provider submission retry or a claim
that the underlying filesystem allocation defect is repaired.

## Checked publication operations

`python managed_operations.py --plan PRIVATE_PLAN --receipt PRIVATE_RECEIPT`
reuses ready processor checkpoints without constructing an inference client.
The version1 plan selects publication, source and SHA-256, checkpoint, optional
CLI manifest for unpublished output, required command names, and approved resource
ceilings (workers, gpu_budget_bytes, headroom_bytes, system_ram_bytes).
`--preflight-only` validates without publication. `--check-script PRIVATE_SCRIPT`
applies the existing pinned Ruff F check without importing/executing that script.
Failure receipts contain only exception class, errno and bounded code locations;
raw exception messages, environment and transcript content are excluded.

## Durable service intent

`managed_service_intent.save_intent` records explicit batch continuation and paused
service names in private operator state. The `managed_service_intent.py --intent
PRIVATE_INTENT --receipt PRIVATE_RECEIPT` entrypoint stops and confirms paused
services before starting the existing checkpoint-aware batch controller. Unknown
service state holds continuation. It submits no inference requests and does not
clear processor uncertainty. Install a user startup/periodic reconciliation unit
with the pinned application and intent paths; after terminal completion persist
continue_batch=false. Speech stays paused until an explicit operator override.

A version1 operation with mode=resume observes/advances the existing checkpoint
under a bounded wait (default300seconds). It holds failed/cancelled/unknown attempts
for explicit reconciliation and retries only observation of existing active IDs
after transient transport failures. It does not create replacement attempts.
Provide the existing connection settings for the managed client. With a verified
CLI manifest, completion publishes through the same guarded publication callback;
without one it returns a ready checkpoint. Completed results are always reused.
