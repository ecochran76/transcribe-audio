#!/usr/bin/env python3
"""
Reviewed maintenance for local transcript review queue state.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from transcript_api import DEFAULT_STATE_DIR, read_json_file

ARCHIVE_APPROVAL_TOKEN = "ARCHIVE_STALE_ROUTE_REVIEWS"
DEFAULT_ARCHIVE_DIR = Path("~/.local/state/transcribe-audio/review-queue-archive")


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def archive_path_for(archive_root: Path, source_path: Path, *, run_id: str) -> Path:
    return archive_root.expanduser().resolve() / run_id / source_path.name


def stale_route_review_plan(
    *,
    state_root: Path = DEFAULT_STATE_DIR,
    archive_root: Path = DEFAULT_ARCHIVE_DIR,
    run_id: Optional[str] = None,
) -> dict[str, Any]:
    runtime_state_root = state_root.expanduser().resolve()
    review_dir = runtime_state_root / "review-queue"
    archive_run_id = run_id or utc_stamp()
    items: list[dict[str, Any]] = []
    for path in sorted(review_dir.glob("*.route-review.json")):
        payload = read_json_file(path)
        route_path_text = str(payload.get("route_decision_path") or "")
        route_path = Path(route_path_text).expanduser() if route_path_text else None
        if route_path is not None and route_path.exists():
            continue
        items.append(
            {
                "id": path.stem.removesuffix(".route-review"),
                "source_path": str(path),
                "archive_path": str(archive_path_for(archive_root, path, run_id=archive_run_id)),
                "route_decision_path": route_path_text,
                "selected_label": payload.get("selected_label") or "",
                "reason": "route decision path is missing",
                "created_at": payload.get("created_at") or "",
            }
        )
    return {
        "schema_version": 1,
        "generated_at": utc_now_iso(),
        "state_dir": str(runtime_state_root),
        "review_queue_dir": str(review_dir),
        "archive_dir": str(archive_root.expanduser().resolve()),
        "run_id": archive_run_id,
        "action": "archive_stale_route_reviews",
        "apply": False,
        "summary": {
            "candidate_count": len(items),
        },
        "items": items,
    }


def apply_stale_archive_plan(plan: dict[str, Any], *, apply: bool = False) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for item in plan.get("items") or []:
        if not isinstance(item, dict):
            continue
        source_path = Path(str(item.get("source_path") or "")).expanduser()
        archive_path = Path(str(item.get("archive_path") or "")).expanduser()
        result = {
            "id": item.get("id") or source_path.stem,
            "source_path": str(source_path),
            "archive_path": str(archive_path),
            "status": "planned",
            "dry_run": not apply,
        }
        if not source_path.exists():
            result["status"] = "missing_source"
            results.append(result)
            continue
        if archive_path.exists():
            result["status"] = "archive_target_exists"
            results.append(result)
            continue
        if apply:
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source_path), str(archive_path))
            result["status"] = "archived"
        results.append(result)
    by_status: dict[str, int] = {}
    for result in results:
        status = str(result.get("status") or "unknown")
        by_status[status] = by_status.get(status, 0) + 1
    audit = {
        **plan,
        "applied_at": utc_now_iso() if apply else "",
        "apply": apply,
        "summary": {
            **(plan.get("summary") if isinstance(plan.get("summary"), dict) else {}),
            "by_status": by_status,
        },
        "results": results,
    }
    return audit


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    output_path = path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return output_path


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan or apply reviewed maintenance for local transcript review queues.")
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE_DIR)
    parser.add_argument("--run-id", help="Stable archive run id. Defaults to a UTC timestamp.")
    parser.add_argument("--plan-output", type=Path, help="Write the dry-run plan JSON here.")
    parser.add_argument("--audit-output", type=Path, help="Write the dry-run/apply audit JSON here.")
    parser.add_argument("--apply", action="store_true", help="Move stale route-review files into the archive directory.")
    parser.add_argument("--approval-token", help=f"Required with --apply. Use {ARCHIVE_APPROVAL_TOKEN}.")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.apply and args.approval_token != ARCHIVE_APPROVAL_TOKEN:
        raise ValueError(f"--apply requires --approval-token {ARCHIVE_APPROVAL_TOKEN}")
    plan = stale_route_review_plan(state_root=args.state_dir, archive_root=args.archive_dir, run_id=args.run_id)
    if args.plan_output:
        plan["plan_path"] = str(write_json(args.plan_output, plan))
    audit = apply_stale_archive_plan(plan, apply=args.apply)
    if args.audit_output:
        audit["audit_path"] = str(write_json(args.audit_output, audit))
    return audit


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    try:
        audit = run(args)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    summary = audit.get("summary") if isinstance(audit.get("summary"), dict) else {}
    print(
        "Review queue maintenance: "
        f"apply={audit.get('apply')} candidates={summary.get('candidate_count', 0)} "
        f"statuses={summary.get('by_status', {})}"
    )
    if audit.get("audit_path"):
        print(f"Audit: {audit['audit_path']}")
    elif audit.get("plan_path"):
        print(f"Plan: {audit['plan_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
