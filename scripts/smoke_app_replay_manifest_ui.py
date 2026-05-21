#!/usr/bin/env python3
"""
Drive the React console through a disposable replay-manifest smoke.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.smoke_app_replay_manifest import DEFAULT_BASE_URL, DEFAULT_STATE_ROOT, run_smoke


DEFAULT_REPORT_DIR = Path("~/.local/state/transcribe-audio/browser-smokes")
UI_SMOKE_JSON_STDOUT_PREFIX = "APP_REPLAY_MANIFEST_UI_SMOKE_JSON="


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Browser-smoke the App Intelligence replay-manifest UI.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Transcript console base URL.")
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT, help="App runtime state root.")
    parser.add_argument("--run-id", default="smoke-replay-manifest-ui-review", help="Disposable run id.")
    parser.add_argument("--session", default="transcript-replay-ui-smoke", help="agent-browser session name.")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR, help="Directory for JSON/screenshot artifacts.")
    parser.add_argument("--cleanup", action="store_true", help="Delete the disposable run after the browser smoke.")
    return parser.parse_args(argv)


def run_agent_browser(session: str, command: list[str], *, timeout: int = 30) -> str:
    proc = subprocess.run(
        ["agent-browser", "--session", session, *command],
        check=True,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    return proc.stdout.strip()


def eval_js(session: str, script: str) -> Any:
    output = run_agent_browser(session, ["eval", script])
    return json.loads(output)


def click_button_js(label: str, *, contains: str) -> str:
    return f"""
(() => {{
  const needle = {json.dumps(contains.lower())};
  const button = [...document.querySelectorAll('button')]
    .find((item) => item.textContent.toLowerCase().includes(needle));
  if (!button) throw new Error({json.dumps(f"Missing {label} button")});
  button.click();
  return button.textContent.trim();
}})()
""".strip()


def run_ui_smoke(
    *,
    base_url: str,
    state_root: Path,
    run_id: str,
    session: str,
    report_dir: Path,
    cleanup: bool,
) -> dict[str, Any]:
    api_smoke = run_smoke(base_url=base_url, state_root=state_root, run_id=run_id, cleanup=False)
    base_url = base_url.rstrip("/")
    run_agent_browser(session, ["open", f"{base_url}/"], timeout=45)
    run_agent_browser(session, ["wait", "1000"])
    click_intelligence = eval_js(session, click_button_js("Intelligence", contains="Intelligence"))
    run_agent_browser(session, ["wait", "1000"])
    click_run = eval_js(session, click_button_js("disposable run", contains=run_id))
    run_agent_browser(session, ["wait", "1000"])
    click_artifact = eval_js(session, click_button_js("replay manifest artifact", contains="PROMPT PACKET JSON"))
    run_agent_browser(session, ["wait", "1000"])
    checks = eval_js(
        session,
        f"""
(() => {{
  const text = document.body.innerText;
  return {{
    hasLiveApi: text.includes('live local API'),
    hasRun: text.includes({json.dumps(run_id)}),
    hasReplayManifest: text.includes('REPLAY MANIFEST'),
    hasPromptPacketJson: text.includes('PROMPT PACKET JSON'),
    hasLoadedMessage: text.includes('no write or external action was executed'),
    hasNoWriteFlag: text.includes('"will_execute_write_bearing_action": false'),
    hasPromptText: text.includes('Disposable replay-manifest smoke prompt.')
  }};
}})()
""".strip(),
    )
    missing = [key for key, value in checks.items() if value is not True]
    report_dir = report_dir.expanduser()
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    screenshot_path = report_dir / f"{stamp}-{run_id}.png"
    run_agent_browser(session, ["screenshot", str(screenshot_path)], timeout=45)
    result = {
        "schema_version": "transcribe-audio.app-replay-manifest-ui-smoke.v1",
        "status": "pass" if not missing else "fail",
        "run_id": run_id,
        "base_url": base_url,
        "session": session,
        "clicked": {
            "intelligence": click_intelligence,
            "run": click_run,
            "artifact": click_artifact,
        },
        "checks": checks,
        "missing_checks": missing,
        "api_smoke": api_smoke,
        "screenshot_path": str(screenshot_path),
        "cleanup": cleanup,
    }
    report_path = report_dir / f"{stamp}-{run_id}.json"
    report_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    result["report_path"] = str(report_path)
    if cleanup:
        run_path = state_root.expanduser() / "app-intelligence-runs" / run_id
        if run_path.exists():
            import shutil

            shutil.rmtree(run_path)
    if missing:
        raise RuntimeError(f"Browser UI smoke failed checks: {', '.join(missing)}")
    return result


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    try:
        result = run_ui_smoke(
            base_url=args.base_url,
            state_root=args.state_root,
            run_id=args.run_id,
            session=args.session,
            report_dir=args.report_dir,
            cleanup=args.cleanup,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"{UI_SMOKE_JSON_STDOUT_PREFIX}{json.dumps(result, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
