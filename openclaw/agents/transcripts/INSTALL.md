# Install Plan: `transcripts` OpenClaw Agent

This directory stores portable OpenClaw workspace files for the `transcripts`
agent. Live runtime state belongs under `~/.openclaw/`, not in this repo.

## Target

- Agent id: `transcripts`
- Runtime workspace: `~/.openclaw/workspace-transcripts`
- Source workspace template: `openclaw/agents/transcripts/workspace/`
- Slack account: `default`
- Slack channel: private channel `oc-transcripts`
- Current live Slack conversation id: `C0B3WDRN38Q`

## OpenClaw Docs Read

- `docs/gateway/config-agents.md`
- `docs/cli/agents.md`
- `docs/channels/channel-routing.md`
- `docs/channels/slack.md`
- `docs/reference/templates/AGENTS.md`

Key constraints from those docs:

- Agent workspace bootstrap files are Markdown files such as `AGENTS.md`,
  `IDENTITY.md`, `TOOLS.md`, `USER.md`, `SOUL.md`, and `HEARTBEAT.md`.
- `openclaw agents add` creates isolated agents with a workspace path.
- Slack has account ids; this install targets `slack/default`.
- Slack channel route bindings should use exact peer ids, not display names.
- Slack private channels usually require `groups:*` scopes and bot membership.

## Safe Install Flow

1. Ensure the private Slack channel `oc-transcripts` exists on the default
   Slack workspace.
2. Resolve its Slack conversation id using OpenClaw directory lookup or Slack
   API. Do not rely on the display name.
3. Dry-run the installer:

```bash
scripts/install_openclaw_transcripts_agent.py --slack-channel-id C0B3WDRN38Q
```

4. Apply the installer:

```bash
scripts/install_openclaw_transcripts_agent.py --apply --slack-channel-id C0B3WDRN38Q
```

The installer copies the Markdown workspace files to
`~/.openclaw/workspace-transcripts`, creates the OpenClaw agent, sets identity,
validates config, and applies the exact Slack channel-peer binding.

Manual equivalent:

```bash
openclaw agents add transcripts \
  --workspace ~/.openclaw/workspace-transcripts \
  --model openai-codex/gpt-5.5 \
  --non-interactive
```

5. Set identity:

```bash
openclaw agents set-identity \
  --agent transcripts \
  --name Transcripts \
  --theme "meeting transcript operations" \
  --emoji transcript
```

6. Bind only the exact Slack private-channel peer on account `default`.
   The binding shape is:

```json5
{
  bindings: [
    {
      agentId: "transcripts",
      match: {
        channel: "slack",
        accountId: "default",
        peer: { kind: "channel", id: "G_OR_C_CHANNEL_ID" },
      },
    },
  ],
}
```

7. Restart or reload OpenClaw if required, then verify:

```bash
openclaw agents list --bindings
openclaw channels status --json
openclaw agent --agent transcripts --message "Report transcript service status"
```

Live validation on 2026-05-11 posted a Slack smoke message in
`oc-transcripts`; `transcripts` replied `TRANSCRIPTS_BINDING_SMOKE_OK`.

## Guardrails

- Do not bind the whole Slack default account to `transcripts`; use the exact
  `oc-transcripts` channel peer id.
- Do not use the `soylei` Slack account for this agent.
- Do not store live memory, sessions, OAuth tokens, or Slack state in this
  repo.
