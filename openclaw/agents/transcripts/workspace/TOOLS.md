# Tools

## Local Repo

- Repo root: `/home/ecochran76/workspace.local/transcribe-audio`
- Primary watcher: `watch_transcriptions.py`
- Transcript sidecars: `*.transcript.json`
- Readouts: `*.readout.json` and `*.readout.md`
- Route decisions: `*.route.json`
- Review queue: `~/.local/state/transcribe-audio/review-queue/`

## Useful Commands

```bash
systemctl --user status transcribe-watch.service
systemctl --user is-active transcribe-watch.service
python route_transcript.py TRANSCRIPT_JSON READOUT_JSON --gws-provenance
python summarize_transcript.py TRANSCRIPT_JSON --provider codex-exec --model gpt-5.5
```

## Context Sources

- Calendar metadata in transcript sidecars is first-class provenance.
- `gws` Calendar and Drive metadata can add read-only provenance.
- Graphiti/OpenClaw memory is advisory until verified against cited sources.
- Odoo, Google Drive deposition, and Graphiti memory writes require explicit
  write authorization and a dry-run preview.

