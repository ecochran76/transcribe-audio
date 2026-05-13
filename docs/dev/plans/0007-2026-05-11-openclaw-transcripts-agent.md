# Plan 0007 | OpenClaw Transcripts Agent

State: CLOSED

Lane: P07

## Scope

Package a dedicated OpenClaw agent named `transcripts` for transcript
operations and bind it to the private Slack channel `oc-transcripts` on the
default Slack account.

## Non-Goals

- Do not bind the entire default Slack account to this agent.
- Do not use the SoyLei Slack account.
- Do not store live OpenClaw sessions, tokens, Slack state, or agent memory in
  this repo.
- Do not enable unattended deposition or external writes through this agent
  without the corresponding P04/P05 write contracts.

## Current State

OpenClaw docs were reviewed for agent config, CLI agent management, channel
routing, Slack behavior, and workspace Markdown templates. Portable workspace
files now exist under `openclaw/agents/transcripts/workspace/`. The installer
at `scripts/install_openclaw_transcripts_agent.py` is dry-run-first and can
copy Markdown files, run the safe `openclaw agents add` and identity commands,
and apply the exact Slack route binding when `--slack-channel-id` is provided.
Live state: private Slack channel `oc-transcripts` was created on the default
Slack tenant with conversation id `C0B3WDRN38Q`, the OpenClaw bot was invited,
and the `transcripts` agent is bound to `slack/default` for that channel peer.

## Work Items

- Done: read OpenClaw docs for workspace file and routing contracts.
- Done: add portable Markdown workspace files for `transcripts`.
- Done: add dry-run-first installer scaffold.
- Done: resolve or create private Slack channel `oc-transcripts` on account
  `default`.
- Done: resolve the exact Slack conversation id for route binding.
- Done: extend installer to apply the exact channel-peer route binding
  idempotently.
- Done: run live install and verify route behavior from the private channel.

## Acceptance Criteria

- Agent files are portable and contain no secrets or raw transcripts.
- Installer dry-run shows all file copies and OpenClaw commands before apply.
- Live OpenClaw config has an agent id `transcripts`.
- Live route binding targets `slack/default` plus the exact
  `oc-transcripts` channel peer id.
- A smoke message in `oc-transcripts` reaches the `transcripts` agent.

## Validation

- `python -m py_compile scripts/install_openclaw_transcripts_agent.py`
- `scripts/install_openclaw_transcripts_agent.py` dry-run
- `openclaw agents list --bindings`
- `openclaw channels status --json`
- Smoke message through the private Slack channel after live binding.

Evidence:

- Slack private channel id: `C0B3WDRN38Q`.
- OpenClaw binding detail: `slack accountId=default peer=channel:C0B3WDRN38Q`.
- Live Slack smoke reply: `TRANSCRIPTS_BINDING_SMOKE_OK`.
