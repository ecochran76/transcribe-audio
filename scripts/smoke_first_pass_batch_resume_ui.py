#!/usr/bin/env python3
"""
Drive the React Review Queue through saved first-pass batch resume status.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.smoke_app_replay_manifest import DEFAULT_BASE_URL, DEFAULT_STATE_ROOT
from scripts.smoke_app_replay_manifest_ui import (
    DEFAULT_REPORT_DIR,
    click_button_js,
    eval_js,
    run_agent_browser,
)


FIRST_PASS_RESUME_UI_SMOKE_JSON_STDOUT_PREFIX = "FIRST_PASS_RESUME_UI_SMOKE_JSON="


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Browser-smoke first-pass batch resume UI.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Transcript console base URL.")
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT, help="App runtime state root.")
    parser.add_argument("--session", default="transcript-first-pass-resume-ui-smoke", help="agent-browser session name.")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR, help="Directory for JSON/screenshot artifacts.")
    parser.add_argument("--cleanup", action="store_true", help="Delete the disposable prepared manifest after the smoke.")
    return parser.parse_args(argv)


def get_json(url: str) -> dict[str, Any]:
    with urlopen(url, timeout=10) as response:
        payload = json.loads(response.read())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object from {url}")
    return payload


def write_disposable_manifest(state_root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = state_root.expanduser() / "first-pass-summary-batches"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"first-pass-summary-prepare-ui-resume-smoke-{stamp}.json"
    payload = {
        "object": "transcribe_audio_auracall_batch_manifest",
        "request_count": 1,
        "dry_run": True,
        "batch": None,
        "batch_payload": {
            "metadata": {"workflow": "transcribe-audio-first-pass-summary"},
            "requests": [
                {
                    "custom_id": "ui-resume-smoke",
                    "metadata": {
                        "outputContract": {"artifactFileName": "first_pass_readout.json"},
                    },
                }
            ],
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def run_smoke(
    *,
    base_url: str,
    state_root: Path,
    session: str,
    report_dir: Path,
    cleanup: bool,
) -> dict[str, Any]:
    base_url = base_url.rstrip("/")
    manifest_path = write_disposable_manifest(state_root)
    manifest_payload = get_json(f"{base_url}/api/review-queue/first-pass-summaries/manifests?limit=5")
    listed = [item for item in manifest_payload.get("items", []) if item.get("manifest") == str(manifest_path)]
    if not listed:
        raise RuntimeError(f"Disposable manifest was not listed by API: {manifest_path}")

    run_agent_browser(session, ["open", f"{base_url}/"], timeout=45)
    run_agent_browser(session, ["wait", "1000"])
    click_review = eval_js(session, click_button_js("Review Queue", contains="Review Queue"))
    run_agent_browser(session, ["wait", "1000"])
    click_manifest = eval_js(
        session,
        f"""
(() => {{
  const target = {json.dumps(str(manifest_path))};
  const button = [...document.querySelectorAll('button.saved-batch-row')]
    .find((item) => item.textContent.includes(target));
  if (!button) throw new Error('Missing disposable saved batch row');
  button.click();
  return button.textContent.trim();
}})()
""".strip(),
    )
    run_agent_browser(session, ["wait", "500"])
    click_status = eval_js(session, click_button_js("Check and materialize", contains="Check and materialize"))
    run_agent_browser(session, ["wait", "1000"])
    checks = eval_js(
        session,
        f"""
(() => {{
  const text = document.body.innerText;
  return {{
    hasReviewQueue: text.includes('Review queue'),
    hasRecentBatches: text.includes('Recent first-pass batches'),
    hasManifest: text.includes({json.dumps(str(manifest_path))}),
    hasPreparedStatus: text.includes('Batch status prepared'),
    hasRequestCount: text.includes('REQUESTS') && text.includes('1 requests'),
    hasPreparedOnly: text.includes('prepared only')
  }};
}})()
""".strip(),
    )
    missing = [key for key, value in checks.items() if value is not True]
    report_dir = report_dir.expanduser()
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    screenshot_path = report_dir / f"{stamp}-first-pass-resume-ui-smoke.png"
    run_agent_browser(session, ["screenshot", str(screenshot_path)], timeout=45)
    result = {
        "schema_version": "transcribe-audio.first-pass-resume-ui-smoke.v1",
        "status": "pass" if not missing else "fail",
        "base_url": base_url,
        "session": session,
        "manifest": str(manifest_path),
        "clicked": {
            "review_queue": click_review,
            "manifest": click_manifest,
            "status": click_status,
        },
        "checks": checks,
        "missing_checks": missing,
        "screenshot_path": str(screenshot_path),
        "cleanup": cleanup,
    }
    report_path = report_dir / f"{stamp}-first-pass-resume-ui-smoke.json"
    report_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    result["report_path"] = str(report_path)
    if cleanup:
        manifest_path.unlink(missing_ok=True)
    if missing:
        raise RuntimeError(f"First-pass resume UI smoke failed checks: {', '.join(missing)}")
    return result


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    try:
        result = run_smoke(
            base_url=args.base_url,
            state_root=args.state_root,
            session=args.session,
            report_dir=args.report_dir,
            cleanup=args.cleanup,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"{FIRST_PASS_RESUME_UI_SMOKE_JSON_STDOUT_PREFIX}{json.dumps(result, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
