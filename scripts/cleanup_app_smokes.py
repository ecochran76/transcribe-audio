#!/usr/bin/env python3
"""
Clean disposable App Intelligence smoke runs and browser-smoke evidence.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional


DEFAULT_STATE_ROOT = Path("~/.local/state/transcribe-audio")
DEFAULT_RUN_PREFIX = "smoke-replay-manifest"
DEFAULT_BROWSER_SMOKE_DIR = DEFAULT_STATE_ROOT / "browser-smokes"
SMOKE_CLEANUP_STDOUT_PREFIX = "APP_SMOKE_CLEANUP_JSON="


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean disposable App Intelligence smoke artifacts.")
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT, help="Runtime state root.")
    parser.add_argument("--run-prefix", default=DEFAULT_RUN_PREFIX, help="Only clean run dirs with this prefix.")
    parser.add_argument("--keep-runs", type=int, default=1, help="Keep the newest matching run dirs.")
    parser.add_argument("--evidence-days", type=int, default=14, help="Keep browser-smoke evidence newer than this many days.")
    parser.add_argument("--keep-evidence", type=int, default=10, help="Keep at least this many newest browser-smoke evidence files.")
    parser.add_argument("--apply", action="store_true", help="Delete selected files/directories. Defaults to dry run.")
    parser.add_argument("--format", choices=("json", "text"), default="json")
    return parser.parse_args(argv)


def stat_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def matching_run_dirs(state_root: Path, prefix: str) -> list[Path]:
    root = state_root / "app-intelligence-runs"
    if not root.exists():
        return []
    return sorted(
        [path for path in root.iterdir() if path.is_dir() and path.name.startswith(prefix)],
        key=stat_mtime,
        reverse=True,
    )


def matching_evidence_files(state_root: Path) -> list[Path]:
    root = state_root / "browser-smokes"
    if not root.exists():
        return []
    return sorted([path for path in root.iterdir() if path.is_file()], key=stat_mtime, reverse=True)


def cleanup_smokes(
    *,
    state_root: Path,
    run_prefix: str,
    keep_runs: int,
    evidence_days: int,
    keep_evidence: int,
    apply: bool,
) -> dict[str, Any]:
    state_root = state_root.expanduser()
    runs = matching_run_dirs(state_root, run_prefix)
    keep_run_count = max(0, keep_runs)
    run_delete = runs[keep_run_count:]
    evidence = matching_evidence_files(state_root)
    cutoff = datetime.now(timezone.utc) - timedelta(days=max(0, evidence_days))
    keep_evidence_count = max(0, keep_evidence)
    evidence_delete: list[Path] = []
    for index, path in enumerate(evidence):
        mtime = datetime.fromtimestamp(stat_mtime(path), tz=timezone.utc)
        if index >= keep_evidence_count and mtime < cutoff:
            evidence_delete.append(path)

    if apply:
        for path in run_delete:
            shutil.rmtree(path, ignore_errors=True)
        for path in evidence_delete:
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    return {
        "schema_version": "transcribe-audio.app-smoke-cleanup.v1",
        "state_root": str(state_root),
        "run_prefix": run_prefix,
        "apply": apply,
        "kept_run_count": min(len(runs), keep_run_count),
        "matched_run_count": len(runs),
        "delete_run_count": len(run_delete),
        "delete_run_paths": [str(path) for path in run_delete],
        "browser_smoke_dir": str(state_root / "browser-smokes"),
        "matched_evidence_count": len(evidence),
        "keep_evidence": keep_evidence_count,
        "evidence_days": evidence_days,
        "delete_evidence_count": len(evidence_delete),
        "delete_evidence_paths": [str(path) for path in evidence_delete],
    }


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    result = cleanup_smokes(
        state_root=args.state_root,
        run_prefix=args.run_prefix,
        keep_runs=args.keep_runs,
        evidence_days=args.evidence_days,
        keep_evidence=args.keep_evidence,
        apply=args.apply,
    )
    if args.format == "json":
        print(f"{SMOKE_CLEANUP_STDOUT_PREFIX}{json.dumps(result, sort_keys=True)}")
    else:
        mode = "apply" if result["apply"] else "dry-run"
        print(
            f"{mode}: delete_runs={result['delete_run_count']} "
            f"delete_evidence={result['delete_evidence_count']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
