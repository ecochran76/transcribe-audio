# AuraCall Legacy Enrichment Handoff

Date: 2026-05-12

## Current Update

The full `summarize_transcript.py --provider openai-compatible --store` run was retried against the same legacy transcript using `/home/ecochran76/.auracall/api.env` and succeeded.

Retry log:

```text
/home/ecochran76/.local/state/transcribe-audio/legacy-enrichment-smoke-auracall-retry-2026-05-12.log
```

Retry outputs:

```text
/home/ecochran76/.transcripts/legacy-artifacts/07/0711bf42d7771e63b44d-2025-07-28 Scott gener8or imPETus SABER.readout.json
/home/ecochran76/.transcripts/legacy-artifacts/07/0711bf42d7771e63b44d-2025-07-28 Scott gener8or imPETus SABER.readout.md
```

Store verification after the retry:

- `documents.kind='readout'` count is now 5.
- The de-duped pending legacy enrichment queue is now 67 items.
- The generated provider metadata used `base_url=http://127.0.0.1:18095/v1` and `model=agent:instant-chatgpt-ecochran76`.

This means AuraCall is no longer failing for this exact full-transcript smoke. The remaining risk is batch variability across other long or malformed legacy transcript sidecars.

## Batch Retry Update

A later bounded three-item batch against the live queue partially succeeded.

Batch queue:

```text
/home/ecochran76/.local/state/transcribe-audio/legacy-enrichment-batch-3-queue-2026-05-12.json
```

Batch logs:

```text
/home/ecochran76/.local/state/transcribe-audio/legacy-enrichment-batch-3-retry-2026-05-12/
```

Results:

- `20250417-142659-Ambient Workshop Recording (2025-04-17) - Non-Verbal Audio` succeeded and stored a readout beside its legacy transcript sidecar.
- `2025-07-17 Shuana Sofia MacGill` succeeded and stored a readout beside its legacy transcript sidecar.
- `2025-06-06 Breakfast with Nacu My recording 9` failed twice through `summarize_transcript.py` with `OpenAI-compatible readout did not return valid JSON`.

Raw diagnostic call for the failed item:

```text
/home/ecochran76/.local/state/transcribe-audio/legacy-enrichment-batch-3-retry-2026-05-12/03-raw-response.json
/home/ecochran76/.local/state/transcribe-audio/legacy-enrichment-batch-3-retry-2026-05-12/03-raw-content.txt
```

The raw content was not malformed JSON. It was provider error text inside an HTTP 200 chat-completions response:

```text
Something went wrong. If this issue persists please contact us through our help center at help.openai.com.
```

Store verification after the partial batch showed 10 readout documents and 63 pending de-duped first-pass legacy readout items. The failure is therefore provider/model execution variability for that transcript, not the prior timeout issue and not a parser issue.

## Context

The legacy transcript import/backfill work now has a de-duped first-pass readout queue:

```bash
python transcript_store.py legacy-enrichment-queue --format commands --provider openai-compatible
```

At the end of Turn 51, the live store had:

- 79 transcript documents.
- 70 legacy rows still marked `legacy_import.needs_enrichment=true`.
- 68 de-duped pending first-pass readout items.
- 60 transcript documents linked to source-recording blobs.

The intended next step was to smoke one first-pass readout before running a larger enrichment batch.

## Failure

The default OpenAI-compatible provider failed first with quota exhaustion:

```text
OpenAI-compatible readout failed (429): insufficient_quota
```

Evidence log:

```text
/home/ecochran76/.local/state/transcribe-audio/legacy-enrichment-smoke-2026-05-12.log
```

The AuraCall environment was then loaded from:

```text
/home/ecochran76/.auracall/api.env
```

Do not copy or commit values from that file. The environment did set `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `AURACALL_BASE_URL`, and `AURACALL_MODEL`.

The AuraCall-compatible request reached the local endpoint but timed out:

```text
HTTPConnectionPool(host='127.0.0.1', port=18095): Read timed out. (read timeout=120.0)
```

Evidence log:

```text
/home/ecochran76/.local/state/transcribe-audio/legacy-enrichment-smoke-auracall-2026-05-12.log
```

The smoke transcript was:

```text
/home/ecochran76/.transcripts/legacy-artifacts/07/0711bf42d7771e63b44d-2025-07-28 Scott gener8or imPETus SABER.transcript.json
```

## Important Follow-Up

`summarize_transcript.py` was patched after the AuraCall timeout so future `requests` failures are reported as clean `TranscriptionError` messages instead of raw stack traces. Tests were added for this behavior.

Validation after that patch:

```bash
.venv/bin/python -m pytest -q
```

Result:

```text
102 passed
```

## Recommended Next Slice

Update: after AuraCall runtime repair in `/home/ecochran76/workspace.local/auracall`,
the original smoke transcript succeeded through the AuraCall OpenAI-compatible
endpoint on 2026-05-12.

Smoke output:

```text
/home/ecochran76/.local/state/transcribe-audio/auracall-smokes/0711bf42d7771e63b44d-2025-07-28 Scott gener8or imPETus SABER.readout.json
/home/ecochran76/.local/state/transcribe-audio/auracall-smokes/0711bf42d7771e63b44d-2025-07-28 Scott gener8or imPETus SABER.readout.md
```

The generated readout was ingested directly with `transcript_store.ingest_artifact`
to avoid a duplicate provider call. The pending de-duped legacy enrichment queue
then dropped from 68 to 67 items, and the Scott/gener8or item was no longer
pending.

Small-batch update: a three-item AuraCall-backed batch was attempted after the
single smoke passed.

Results:

- `20250417-142659-Ambient Workshop Recording (2025-04-17) - Non-Verbal Audio`
  succeeded and inserted readout id `04499cb40f6e3cc692df`.
- `2025-07-09 Lululemon Summary and` succeeded and inserted readout id
  `603f042617e1e01dcca2`.
- `2024-08-02 Chris Craig ISURF` initially failed after AuraCall returned
  content because `summarize_transcript.py` could not parse the response as
  valid JSON. After adding bounded JSON extraction for wrapped/fenced model
  output, the item succeeded and inserted readout id `9c6803cf743c2ea723fd`.

Batch logs and summary are under:

```text
/home/ecochran76/.local/state/transcribe-audio/auracall-batch-2026-05-12/
```

The batch logs were redacted after creation because the captured command line
included the API key.

Second small-batch update: a five-item batch exposed an AuraCall-side transport
problem for long browser-backed readouts.

- A transcript excerpt-budget workaround was tried and then reverted. The
  readout caller should continue sending the full transcript; AuraCall must
  preserve caller capability instead of forcing downstream truncation.
- The OpenAI-compatible readout parser now rejects JSON objects that do not
  contain actual readout content, preventing echoed prompt/input JSON from
  being stored as empty readouts.
- Four empty readout rows created during the first 90k-budget attempt were
  removed from the store.

Successful second-batch readouts during the workaround experiment:

- `2025-06-06 Breakfast with Nacu My recording 9`
- `2025-07-31 Nacu Breakfast My recording 17`
- `2025-04-24 Nacu Meeting USDA Grant and SoyLei Matters`
- `2025-07-29 Dr Warmbe Meniscus Tear consult`

Still pending:

- `2025-04-17 Nacu Eric Call SoyLei SBIR Matters` returned an empty response on
  one attempt and then hit the 300 second client timeout on retry. AuraCall had
  no recent stuck runtime runs afterward.

The pending de-duped queue now reports 59 items.

Do not start the full enrichment batch yet; handle the remaining SBIR item
separately, then run another bounded batch.

Recommended order:

1. Fix AuraCall so OpenAI-compatible browser-backed requests can carry large
   prompts without composer truncation or HTTP 200 empty responses.
2. Use the AuraCall env from `/home/ecochran76/.auracall/api.env`.
3. Re-run only `2025-04-17 Nacu Eric Call SoyLei SBIR Matters` with the full
   transcript after the AuraCall fix lands.
4. Keep `--timeout 300` for browser-backed AuraCall calls.

The product fix belongs in AuraCall request transport and failure reporting, not
in a transcript-length downgrade inside this repo.

Final update:

- The transcript-length workaround was removed from this repo.
- AuraCall now preserves OpenAI-compatible `response_format`, carries system
  instructions into browser execution, spills oversized browser prompts to
  request attachments, returns HTTP 502 for failed chat-completions runs, and
  detects incomplete ChatGPT JSON-object responses.
- The pending SBIR transcript was retried with the full transcript.
  `agent:instant-chatgpt-soylei` failed honestly with HTTP 502 because ChatGPT
  did not finish parseable JSON; `agent:pro-extended-chatgpt-soylei` succeeded.
- The SBIR readout is now written at
  `/home/ecochran76/.transcripts/legacy-artifacts/28/28d268e46f590765c413-2025-04-17 Nacu Eric Call SoyLei SBIR Matters.readout.json`.
- The de-duped pending legacy enrichment queue now reports 58 items.

2026-05-13 batch-queue update:

- Transcribe Audio now has a first-class AuraCall response-batch client:
  `scripts/auracall_legacy_enrichment_batch.py`.
- The intended default model for legacy readout bursts is
  `agent:pro-extended-chatgpt-soylei-transcripts`.
- The scoped client env is user-scoped at
  `/home/ecochran76/.local/state/transcribe-audio/auracall-transcripts.env`.
- `enqueue` submits selected legacy readouts as one AuraCall response batch;
  AuraCall owns concurrency and browser interaction rate limiting.
- `status --materialize --store` reads completed AuraCall responses and writes
  `*.readout.json` and `*.readout.md` through the same readout materialization
  path as `summarize_transcript.py`.
- A one-item live enqueue created and completed
  `batch_0db1883c7905471c83d807411cfdee33` as
  `resp_1a4b0915303848a6ab68a48e286e563f`.
- Materialization wrote and stored:
  `/home/ecochran76/.transcripts/legacy-artifacts/29/29ed3d64cca92a7cf5f5-2025-08-15 Dr Stefl Knee Replacement Consult.readout.json`.
- The de-duped pending legacy enrichment queue now reports 57 items.
- The provider project ensure call for ChatGPT project `Transcripts` currently
  fails with `button-missing`; the registry agent is configured with
  `projectName=Transcripts`, but AuraCall still needs the ChatGPT project
  creation/binding selector repair before this is fully project-bound in the
  provider workbench.

2026-05-13 project-binding update:

- AuraCall repaired the ChatGPT project-create confirm selector drift.
- `POST /v1/projects/ensure` now creates/finds ChatGPT project `Transcripts`
  and binds `agent:pro-extended-chatgpt-soylei-transcripts`.
- Provider project id:
  `g-p-6a04628762ac8191894b16cfaddfd126`.
- A second ensure call returned `status=found`, so future setup runs should be
  idempotent.

2026-05-13 scoped-client readiness update:

- The scoped client env still exists at
  `/home/ecochran76/.local/state/transcribe-audio/auracall-transcripts.env`.
- The running AuraCall service advertises
  `agent:pro-extended-chatgpt-soylei-transcripts` to that scoped key.
- `pnpm run smoke:scoped-client-env -- /home/ecochran76/.local/state/transcribe-audio/auracall-transcripts.env --prompt 'Reply exactly: auracall transcribe env ok' --expect-output 'auracall transcribe env ok' --timeout-ms 180000`
  passed through the live browser-backed SoyLei Pro Extended transcript agent.
- Response id:
  `resp_45008e83347940909bcdba697b91fa2c`.
- Readback status was `completed` with output
  `auracall transcribe env ok`.

Use this path for the next bounded legacy-enrichment batch. Do not reintroduce
caller-side transcript truncation; AuraCall owns large prompt transport,
project binding, queueing, and browser interaction rate limiting.
