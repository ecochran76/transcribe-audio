#!/usr/bin/env python3
"""
Browser-smoke the Library deep-link and workspace-link copy flow.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.smoke_app_replay_manifest import DEFAULT_BASE_URL
from scripts.smoke_app_replay_manifest_ui import DEFAULT_REPORT_DIR, click_button_js


LIBRARY_SHARE_UI_SMOKE_JSON_STDOUT_PREFIX = "LIBRARY_SHARE_UI_SMOKE_JSON="


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Browser-smoke Library deep-link and share URL behavior.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Transcript console base URL.")
    parser.add_argument("--session", default="transcript-library-share-ui-smoke", help="agent-browser session name.")
    parser.add_argument("--profile", type=Path, default=None, help="Optional browser profile path. Defaults to an isolated profile under report-dir.")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR, help="Directory for JSON/screenshot artifacts.")
    parser.add_argument("--query", default="Cara", help="Library search query for the deep-link smoke.")
    parser.add_argument("--kind", default="readout", help="Library kind filter for the deep-link smoke.")
    parser.add_argument("--workflow", default="context", help="Conversation workspace tab for the deep-link smoke.")
    parser.add_argument("--viewport", default="1280x820", help="Viewport size as WIDTHxHEIGHT.")
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


def run_smoke(
    *,
    base_url: str,
    session: str,
    profile: Path,
    report_dir: Path,
    query: str,
    kind: str,
    workflow: str,
    viewport: tuple[int, int],
) -> dict[str, Any]:
    agent_browser_bin = resolve_agent_browser_bin()
    base_url = base_url.rstrip("/")
    report_dir = report_dir.expanduser()
    report_dir.mkdir(parents=True, exist_ok=True)
    profile = profile.expanduser()
    profile.mkdir(parents=True, exist_ok=True)

    params = urlencode({"kind": kind, "q": query, "conversation": "1", "workflow": workflow})
    deep_link_url = f"{base_url}/?{params}"
    width, height = viewport

    run_agent_browser(agent_browser_bin, session, profile, ["set", "viewport", str(width), str(height)])
    run_agent_browser(agent_browser_bin, session, profile, ["open", deep_link_url])
    run_agent_browser(agent_browser_bin, session, profile, ["wait", "1800"])
    modal_checks = eval_js(
        agent_browser_bin,
        session,
        profile,
        f"""
(() => {{
  const text = document.body.innerText;
  const lowerText = text.toLowerCase();
  const url = new URL(window.location.href);
  const activeFilterText = [...document.querySelectorAll('.library-kind-controls button[aria-pressed="true"], .filter-card button[aria-pressed="true"]')]
    .map((item) => item.textContent.toLowerCase())
    .join(' ');
  return {{
    hasLibrary: text.includes('Transcript library'),
    hasConversationWorkspace: lowerText.includes('conversation workspace'),
    hasWorkflow: lowerText.includes('context workbench'),
    queryValue: document.querySelector('.library-search input, .global-search input')?.value || '',
    kindFilterPressed: activeFilterText.includes({json.dumps(kind.lower())})
      || ({json.dumps(kind)} === 'readout' && activeFilterText.includes('summaries')),
    activeFilterText,
    urlKind: url.searchParams.get('kind'),
    urlQuery: url.searchParams.get('q'),
    urlConversation: url.searchParams.get('conversation'),
    urlWorkflow: url.searchParams.get('workflow')
  }};
}})()
""".strip(),
    )

    eval_js(
        agent_browser_bin,
        session,
        profile,
        """
(() => {
  const button = document.querySelector('button[aria-label="Close conversation workspace"]');
  if (!button) throw new Error('Missing close conversation workspace button');
  button.click();
  return true;
})()
""".strip(),
    )
    run_agent_browser(agent_browser_bin, session, profile, ["wait", "500"])
    click_share = eval_js(agent_browser_bin, session, profile, click_button_js("Copy workspace link", contains="Copy workspace link"))
    run_agent_browser(agent_browser_bin, session, profile, ["wait", "500"])
    share_checks = eval_js(
        agent_browser_bin,
        session,
        profile,
        f"""
(() => {{
  const text = document.body.innerText;
  const input = document.querySelector('input[aria-label="Current workspace link"]');
  const value = input?.value || '';
  return {{
    hasShareButton: text.includes('Copy workspace link'),
    hasCopiedStatus: text.includes('Copied current workspace link.'),
    hasManualFallback: text.includes('Clipboard blocked. Select the link below to copy it manually.'),
    hasWorkspaceLinkInput: Boolean(input),
    workspaceLinkValue: value,
    workspaceLinkHasKind: value.includes('kind={kind}'),
    workspaceLinkHasQuery: value.includes('q={query}'),
    workspaceLinkHasSelected: value.includes('selected=')
  }};
}})()
""".strip(),
    )
    share_ok = share_checks["hasCopiedStatus"] or (
        share_checks["hasManualFallback"]
        and share_checks["hasWorkspaceLinkInput"]
        and share_checks["workspaceLinkHasKind"]
        and share_checks["workspaceLinkHasQuery"]
        and share_checks["workspaceLinkHasSelected"]
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    screenshot_path = report_dir / f"{stamp}-library-share-ui-smoke.png"
    run_agent_browser(agent_browser_bin, session, profile, ["screenshot", str(screenshot_path)], timeout=45)

    checks = {
        **{f"modal_{key}": value for key, value in modal_checks.items()},
        **{f"share_{key}": value for key, value in share_checks.items()},
        "share_flow_ok": share_ok,
    }
    expected = {
        "modal_hasLibrary": True,
        "modal_hasConversationWorkspace": True,
        "modal_hasWorkflow": True,
        "modal_queryValue": query,
        "modal_kindFilterPressed": True,
        "modal_urlKind": kind,
        "modal_urlQuery": query,
        "modal_urlConversation": "1",
        "modal_urlWorkflow": workflow,
        "share_hasShareButton": True,
        "share_flow_ok": True,
    }
    missing = [key for key, expected_value in expected.items() if checks.get(key) != expected_value]
    result = {
        "schema_version": "transcribe-audio.library-share-ui-smoke.v1",
        "status": "pass" if not missing else "fail",
        "base_url": base_url,
        "deep_link_url": deep_link_url,
        "session": session,
        "profile": str(profile),
        "viewport": {"width": width, "height": height},
        "clicked": {"share": click_share},
        "checks": checks,
        "expected": expected,
        "missing_checks": missing,
        "screenshot_path": str(screenshot_path),
    }
    report_path = report_dir / f"{stamp}-library-share-ui-smoke.json"
    report_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    result["report_path"] = str(report_path)
    if missing:
        raise RuntimeError(f"Library share UI smoke failed checks: {', '.join(missing)}")
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
            workflow=args.workflow,
            viewport=parse_viewport(args.viewport),
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"{LIBRARY_SHARE_UI_SMOKE_JSON_STDOUT_PREFIX}{json.dumps(result, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
