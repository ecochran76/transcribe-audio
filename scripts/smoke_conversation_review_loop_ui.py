#!/usr/bin/env python3
"""
Browser-smoke the M1 dogfoodable conversation review loop.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import urlencode
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.smoke_app_replay_manifest import DEFAULT_BASE_URL
from scripts.smoke_app_replay_manifest_ui import DEFAULT_REPORT_DIR


CONVERSATION_REVIEW_LOOP_SMOKE_JSON_STDOUT_PREFIX = "CONVERSATION_REVIEW_LOOP_SMOKE_JSON="


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Browser-smoke the conversation review loop workspace.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Transcript console base URL.")
    parser.add_argument("--session", default="transcript-conversation-review-loop-smoke", help="agent-browser session name.")
    parser.add_argument("--profile", type=Path, default=None, help="Optional browser profile path.")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR, help="Directory for JSON/screenshot artifacts.")
    parser.add_argument("--query", default="Tempo", help="Library search query for a contextual conversation.")
    parser.add_argument("--kind", default="contextual_readout", help="Library kind filter.")
    parser.add_argument("--viewport", default="1360x860", help="Viewport size as WIDTHxHEIGHT.")
    return parser.parse_args(argv)


def resolve_agent_browser_bin() -> str:
    configured = os.environ.get("AGENT_BROWSER_BIN", "").strip()
    candidates = [
        configured,
        shutil.which("agent-browser") or "",
        str(Path("~/.local/bin/agent-browser").expanduser()),
        str(Path("~/.local/share/pnpm/agent-browser").expanduser()),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    raise FileNotFoundError("agent-browser is not on PATH; set AGENT_BROWSER_BIN or install ~/.local/bin/agent-browser.")


def parse_viewport(value: str) -> tuple[int, int]:
    width, height = value.lower().split("x", 1)
    return int(width), int(height)


def run_agent_browser(agent_browser_bin: str, session: str, profile: Path, command: list[str], *, timeout: int = 45) -> str:
    proc = subprocess.run(
        [agent_browser_bin, "--profile", str(profile), "--session", session, *command],
        check=True,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    return proc.stdout.strip()


def eval_js(agent_browser_bin: str, session: str, profile: Path, script: str) -> Any:
    output = run_agent_browser(agent_browser_bin, session, profile, ["eval", script])
    return json.loads(output)


def click_tab(agent_browser_bin: str, session: str, profile: Path, label: str) -> None:
    eval_js(
        agent_browser_bin,
        session,
        profile,
        f"""
(() => {{
  const button = [...document.querySelectorAll('.workflow-view-tabs button')]
    .find((item) => item.textContent.trim() === {json.dumps(label)});
  if (!button) throw new Error(`Missing workflow tab: {label}`);
  button.click();
  return true;
}})()
""".strip(),
    )
    run_agent_browser(agent_browser_bin, session, profile, ["wait", "350"])


def first_conversation_id(base_url: str, *, kind: str, query: str) -> str:
    params = urlencode({"kind": kind, "query": query, "limit": "1"})
    with urlopen(f"{base_url.rstrip('/')}/api/conversations?{params}", timeout=20) as response:
        payload = json.loads(response.read())
    items = payload.get("items") if isinstance(payload, dict) else []
    first = items[0] if items else {}
    representative = first.get("representative") if isinstance(first.get("representative"), dict) else {}
    return str(representative.get("id") or "")


def collect_checks(agent_browser_bin: str, session: str, profile: Path) -> dict[str, Any]:
    return eval_js(
        agent_browser_bin,
        session,
        profile,
        """
(() => {
  const text = document.body.innerText;
  const lowerText = text.toLowerCase();
  const activeTab = [...document.querySelectorAll('.workflow-view-tabs button')]
    .find((button) => button.getAttribute('aria-pressed') === 'true')?.textContent.trim() || '';
  return {
    activeTab,
    hasConversationWorkspace: lowerText.includes('conversation workspace'),
    hasSourceAudio: Boolean(document.querySelector('.conversation-rail audio[src*="/api/blobs/"]')),
    hasTranscriptTurns: document.querySelectorAll('.transcript-turn').length > 0,
    hasSummaryReady: text.includes('Summary ready') || text.includes('First-pass summary'),
    hasSpeakerReview: text.includes('pending assignments') || text.includes('Speaker assignment'),
    hasContextWorkbench: text.includes('Included provenance') && text.includes('Excluded provenance'),
    hasFinalPreview: lowerText.includes('deposition and memory preview'),
    hasMemoryCandidates: lowerText.includes('memory candidate'),
    hasQueuePreviewButton: text.includes('Queue preview review'),
    urlConversation: new URL(window.location.href).searchParams.get('conversation'),
    urlWorkflow: new URL(window.location.href).searchParams.get('workflow')
  };
})()
""".strip(),
    )


def run_smoke(
    *,
    base_url: str,
    session: str,
    profile: Path,
    report_dir: Path,
    query: str,
    kind: str,
    viewport: tuple[int, int],
) -> dict[str, Any]:
    agent_browser_bin = resolve_agent_browser_bin()
    base_url = base_url.rstrip("/")
    report_dir = report_dir.expanduser()
    report_dir.mkdir(parents=True, exist_ok=True)
    profile = profile.expanduser()
    profile.mkdir(parents=True, exist_ok=True)
    width, height = viewport
    selected_id = first_conversation_id(base_url, kind=kind, query=query)
    params = urlencode(
        {
            "kind": kind,
            "q": query,
            "conversation": "1",
            "workflow": "transcript",
            "selected": selected_id,
        }
    )
    deep_link_url = f"{base_url}/?{params}"

    run_agent_browser(agent_browser_bin, session, profile, ["set", "viewport", str(width), str(height)])
    run_agent_browser(agent_browser_bin, session, profile, ["open", deep_link_url])
    run_agent_browser(agent_browser_bin, session, profile, ["wait", "2200"])
    checks: dict[str, Any] = {}
    checks.update({f"transcript_{key}": value for key, value in collect_checks(agent_browser_bin, session, profile).items()})
    for label, prefix in [
        ("First-pass summary", "summary"),
        ("Speakers", "speakers"),
        ("Context workbench", "context"),
        ("Final readout", "final"),
    ]:
        click_tab(agent_browser_bin, session, profile, label)
        checks.update({f"{prefix}_{key}": value for key, value in collect_checks(agent_browser_bin, session, profile).items()})

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    screenshot_path = report_dir / f"{stamp}-conversation-review-loop-smoke.png"
    run_agent_browser(agent_browser_bin, session, profile, ["screenshot", str(screenshot_path)], timeout=45)
    expected = {
        "transcript_hasConversationWorkspace": True,
        "transcript_hasSourceAudio": True,
        "transcript_hasTranscriptTurns": True,
        "summary_hasSummaryReady": True,
        "speakers_hasSpeakerReview": True,
        "context_hasContextWorkbench": True,
        "final_hasFinalPreview": True,
        "final_hasMemoryCandidates": True,
        "final_hasQueuePreviewButton": True,
        "final_urlConversation": "1",
        "final_urlWorkflow": "output",
    }
    missing = [key for key, expected_value in expected.items() if checks.get(key) != expected_value]
    result = {
        "schema_version": "transcribe-audio.conversation-review-loop-smoke.v1",
        "status": "pass" if not missing else "fail",
        "base_url": base_url,
        "deep_link_url": deep_link_url,
        "session": session,
        "profile": str(profile),
        "viewport": {"width": width, "height": height},
        "checks": checks,
        "expected": expected,
        "missing_checks": missing,
        "screenshot_path": str(screenshot_path),
    }
    report_path = report_dir / f"{stamp}-conversation-review-loop-smoke.json"
    report_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    result["report_path"] = str(report_path)
    if missing:
        raise RuntimeError(f"Conversation review loop smoke failed checks: {', '.join(missing)}")
    return result


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    default_profile = args.report_dir.expanduser() / "profiles" / args.session
    try:
        result = run_smoke(
            base_url=args.base_url,
            session=args.session,
            profile=args.profile or default_profile,
            report_dir=args.report_dir,
            query=args.query,
            kind=args.kind,
            viewport=parse_viewport(args.viewport),
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"{CONVERSATION_REVIEW_LOOP_SMOKE_JSON_STDOUT_PREFIX}{json.dumps(result, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
