#!/usr/bin/env python3
"""
Create a disposable App Intelligence run and verify replay artifact reads.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import quote
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app_intelligence_ledger


DEFAULT_BASE_URL = "http://127.0.0.1:18876"
DEFAULT_STATE_ROOT = Path("~/.local/state/transcribe-audio")
SMOKE_JSON_STDOUT_PREFIX = "APP_REPLAY_MANIFEST_SMOKE_JSON="


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke the live App Intelligence replay-manifest API.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Transcript API base URL.")
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT, help="App runtime state root.")
    parser.add_argument("--run-id", default="", help="Disposable run id to create or reuse.")
    parser.add_argument("--cleanup", action="store_true", help="Delete the disposable run after the smoke.")
    parser.add_argument("--format", choices=("json", "text"), default="json")
    return parser.parse_args(argv)


def get_json(url: str) -> dict[str, Any]:
    with urlopen(url, timeout=10) as response:
        payload = json.loads(response.read())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object from {url}")
    return payload


def make_disposable_run(*, state_root: Path, run_id: str) -> dict[str, Any]:
    app_intelligence_ledger.create_run(
        state_root=state_root,
        workflow="app_replay_manifest_smoke",
        purpose="Disposable replay manifest smoke; safe to delete.",
        document_id="smoke-document",
        run_id=run_id,
        created_by="smoke_app_replay_manifest.py",
    )
    run = app_intelligence_ledger.response_for_run(state_root=state_root, run_id=run_id, event_limit=1)["run"]
    state = run.get("state") if isinstance(run.get("state"), dict) else {}
    app_intelligence_ledger.update_run_json(
        state_root=state_root,
        run_id=run_id,
        updates={
            "status": "running",
            "phase": "session_started",
            "state": {
                **state,
                "active_codex_thread_id": None,
                "latest_turn_id": None,
                "app_server": {
                    "transport": "smoke",
                    "model_turn_started": False,
                    "started_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                },
            },
        },
    )
    return app_intelligence_ledger.prepare_model_turn_packet(
        state_root=state_root,
        run_id=run_id,
        task="app_replay_manifest_smoke",
        route={
            "task": "app_replay_manifest_smoke",
            "provider": "codex-app-server",
            "model": "smoke",
            "requires_ledger": True,
        },
        document={
            "id": "smoke-document",
            "title": "Disposable Replay Manifest Smoke",
            "kind": "smoke",
            "text_preview": "This is a non-sensitive disposable prompt artifact for replay-manifest smoke testing.",
        },
        prompt_text=(
            "Disposable replay-manifest smoke prompt.\n"
            "Do not send this prompt to a model; it exists only to verify registered artifact reads.\n"
        ),
        approval_token=app_intelligence_ledger.MODEL_TURN_PREFLIGHT_TOKEN,
    )


def run_smoke(*, base_url: str, state_root: Path, run_id: str, cleanup: bool) -> dict[str, Any]:
    state_root = state_root.expanduser()
    base_url = base_url.rstrip("/")
    if not run_id:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"smoke-replay-manifest-{stamp}"
    run_path = state_root / "app-intelligence-runs" / run_id
    if run_path.exists():
        shutil.rmtree(run_path)

    packet = make_disposable_run(state_root=state_root, run_id=run_id)
    manifest = get_json(f"{base_url}/api/intelligence/runs/{quote(run_id)}/replay-manifest")
    artifacts = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), list) else []
    if len(artifacts) < 2:
        raise RuntimeError(f"Expected at least two replay artifacts, got {len(artifacts)}.")
    readable = [item for item in artifacts if isinstance(item, dict) and item.get("can_read_via_artifact_endpoint")]
    if not readable:
        raise RuntimeError("Replay manifest did not expose a readable registered artifact.")
    opened = get_json(
        f"{base_url}/api/intelligence/runs/{quote(run_id)}/artifacts?path={quote(str(readable[0].get('path') or ''), safe='')}"
    )
    if opened.get("will_execute_write_bearing_action") is not False:
        raise RuntimeError("Registered artifact read did not return the expected no-write flag.")
    result = {
        "schema_version": "transcribe-audio.app-replay-manifest-smoke.v1",
        "status": "pass",
        "run_id": run_id,
        "run_path": str(run_path),
        "packet_id": packet.get("packet", {}).get("packet_id"),
        "manifest_artifact_count": manifest.get("artifact_count"),
        "opened_artifact_role": readable[0].get("artifact_role"),
        "opened_relative_path": opened.get("relative_path"),
        "opened_artifact_type": opened.get("artifact_type"),
        "will_execute_external_action": opened.get("will_execute_external_action"),
        "will_execute_write_bearing_action": opened.get("will_execute_write_bearing_action"),
        "cleanup": cleanup,
    }
    if cleanup:
        shutil.rmtree(run_path, ignore_errors=True)
    return result


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    try:
        result = run_smoke(
            base_url=args.base_url,
            state_root=args.state_root,
            run_id=args.run_id,
            cleanup=args.cleanup,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if args.format == "json":
        print(f"{SMOKE_JSON_STDOUT_PREFIX}{json.dumps(result, sort_keys=True)}")
    else:
        print(f"status={result['status']} run_id={result['run_id']} opened={result['opened_relative_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
