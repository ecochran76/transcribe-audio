# P04 Provenance Calibration Fixtures

This directory contains repo-safe materials for Plan 0011 calibration. Keep
private reviewed manifests and reports out of the repo.

## Local State Contract

Reviewed live manifests belong under:

```text
~/.local/state/transcribe-audio/p04-calibration/manifests/
```

Sanitized evaluation reports belong under:

```text
~/.local/state/transcribe-audio/p04-calibration/reports/
```

Repo fixtures may include schema files, fictional examples, and sanitized
summaries. Do not commit raw transcript text, raw audio, private Drive document
content, credentials, OAuth tokens, or unreviewed private meeting artifacts.

## Manifest Shape

Use `manifest.schema.json` as the durable schema reference. Each manifest is a
JSON object with:

- `schema_version`: currently `1`.
- `manifest_id`: stable identifier for the reviewed corpus file.
- `review_status`: `accepted`, `draft`, or `rejected`.
- `cases`: reviewed meetings or synthetic regression cases.

Each case contains redacted `transcript` and `readout` objects plus reviewed
source `decisions`. The transcript object may contain calendar metadata and
participants, but must not contain raw body fields such as `transcript_text`,
`text`, `raw_transcript`, `utterances`, or `words`.

Each decision evaluates one source for one route/readout context. Use
`expected: "include"` when the source should support contextual reread, and
`expected: "exclude"` when it should remain advisory or irrelevant. Reports list
false positives and false negatives without copying source snippets.

## Evaluation

Run the harness against local reviewed manifests:

```bash
.venv/bin/python scripts/evaluate_provenance_calibration.py \
  --manifest-dir ~/.local/state/transcribe-audio/p04-calibration/manifests \
  --output ~/.local/state/transcribe-audio/p04-calibration/reports/p04-source-quality-v1.json \
  --fail-on-mismatch \
  --require-decision-count 12 \
  --require-source-families 4
```

Use the synthetic fixture for repo-local smoke tests only:

```bash
.venv/bin/python scripts/evaluate_provenance_calibration.py \
  docs/dev/fixtures/p04-calibration/synthetic-manifest.json \
  --fail-on-mismatch \
  --require-decision-count 12 \
  --require-source-families 4
```
