#!/usr/bin/env python3
"""Dry-run-first installer for the OpenClaw transcripts agent."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_WORKSPACE = REPO_ROOT / "openclaw" / "agents" / "transcripts" / "workspace"
DEFAULT_WORKSPACE = Path.home() / ".openclaw" / "workspace-transcripts"
MARKDOWN_FILES = ("AGENTS.md", "IDENTITY.md", "TOOLS.md", "USER.md", "SOUL.md", "HEARTBEAT.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--agent-id", default="transcripts")
    parser.add_argument("--model", default="openai-codex/gpt-5.5")
    parser.add_argument("--slack-account", default="default")
    parser.add_argument("--slack-channel-id", help="Resolved Slack conversation id for oc-transcripts")
    parser.add_argument("--apply", action="store_true", help="copy files and run safe OpenClaw commands")
    parser.add_argument("--force", action="store_true", help="overwrite existing Markdown workspace files")
    return parser.parse_args()


def run_command(args: list[str], *, apply: bool) -> None:
    print("+", " ".join(args))
    if apply:
        subprocess.run(args, check=True)


def copy_workspace(target: Path, *, apply: bool, force: bool) -> None:
    for name in MARKDOWN_FILES:
        source = SOURCE_WORKSPACE / name
        dest = target / name
        print(f"copy {source} -> {dest}")
        if not apply:
            continue
        target.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.read_bytes() == source.read_bytes():
            continue
        if dest.exists() and not force:
            raise SystemExit(f"Refusing to overwrite existing file without --force: {dest}")
        shutil.copy2(source, dest)


def print_binding_patch(agent_id: str, slack_account: str, slack_channel_id: str | None) -> None:
    channel_id = slack_channel_id or "G_OR_C_CHANNEL_ID"
    print("\nRoute binding patch shape:")
    print("{")
    print("  bindings: [")
    print("    {")
    print(f'      agentId: "{agent_id}",')
    print("      match: {")
    print('        channel: "slack",')
    print(f'        accountId: "{slack_account}",')
    print(f'        peer: {{ kind: "channel", id: "{channel_id}" }},')
    print("      },")
    print("    },")
    print("  ],")
    print("}")
    if not slack_channel_id:
        print("\nResolve oc-transcripts to a Slack conversation id before applying the binding.")


def apply_binding(agent_id: str, slack_account: str, slack_channel_id: str | None, *, apply: bool) -> None:
    if not slack_channel_id:
        return

    config_path = Path.home() / ".openclaw" / "openclaw.json"
    print(f"update OpenClaw route binding in {config_path}")
    if not apply:
        return

    backup = config_path.with_name(f"openclaw.json.pre-{agent_id}-binding-{int(time.time())}.bak")
    shutil.copy2(config_path, backup)

    config = json.loads(config_path.read_text())
    bindings = config.setdefault("bindings", [])
    filtered = []
    for binding in bindings:
        match = binding.get("match") or {}
        peer = match.get("peer") or {}
        if binding.get("agentId") == agent_id:
            continue
        if match.get("channel") == "slack" and peer.get("id") == slack_channel_id:
            continue
        filtered.append(binding)

    filtered.append(
        {
            "agentId": agent_id,
            "match": {
                "channel": "slack",
                "accountId": slack_account,
                "peer": {"kind": "channel", "id": slack_channel_id},
            },
        }
    )
    config["bindings"] = filtered

    slack = config.setdefault("channels", {}).setdefault("slack", {})
    slack["dangerouslyAllowNameMatching"] = False
    slack.setdefault("channels", {})[slack_channel_id] = {
        "enabled": True,
        "requireMention": False,
        "users": ["UEGM25PMG"],
    }

    config_path.write_text(json.dumps(config, indent=2) + "\n")
    subprocess.run(["openclaw", "config", "validate"], check=True)
    print(f"backup={backup}")


def main() -> int:
    args = parse_args()
    workspace = args.workspace.expanduser().resolve()

    if not SOURCE_WORKSPACE.exists():
        raise SystemExit(f"Missing source workspace: {SOURCE_WORKSPACE}")

    print(f"mode={'apply' if args.apply else 'dry-run'}")
    print(f"source_workspace={SOURCE_WORKSPACE}")
    print(f"target_workspace={workspace}")
    copy_workspace(workspace, apply=args.apply, force=args.force)

    run_command(
        [
            "openclaw",
            "agents",
            "add",
            args.agent_id,
            "--workspace",
            str(workspace),
            "--model",
            args.model,
            "--non-interactive",
        ],
        apply=args.apply,
    )
    run_command(
        [
            "openclaw",
            "agents",
            "set-identity",
            "--agent",
            args.agent_id,
            "--name",
            "Transcripts",
            "--theme",
            "meeting transcript operations",
            "--emoji",
            "transcript",
        ],
        apply=args.apply,
    )

    print_binding_patch(args.agent_id, args.slack_account, args.slack_channel_id)
    apply_binding(args.agent_id, args.slack_account, args.slack_channel_id, apply=args.apply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
