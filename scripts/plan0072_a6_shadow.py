#!/usr/bin/env python3
"""Operate the bounded, zero-effect Plan 0072 A6 campaign ledger."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from identity_review_workflow import IdentityReviewWorkflow  # noqa: E402
from identity_evidence_supervisor import IdentityEvidenceSupervisor  # noqa: E402
from identity_shadow_campaign import (  # noqa: E402
    activate_shadow_campaign,
    finalize_shadow_campaign,
    preview_shadow_campaign,
    record_shadow_case,
    register_new_arrival,
    replay_shadow_campaign,
)


def _read_json(path: Path) -> Any:
    with path.expanduser().open(encoding="utf-8") as stream:
        return json.load(stream)


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, default=str))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plan 0072 A6 private shadow ledger; preview is the only read-only command."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    preview_parser = commands.add_parser("preview")
    preview_parser.add_argument("candidates", type=Path)
    preview_parser.add_argument("--activated-at", required=True)

    activate_parser = commands.add_parser("activate")
    activate_parser.add_argument("preview", type=Path)
    activate_parser.add_argument("--expected-preview-sha256", required=True)
    activate_parser.add_argument("--reviewed-at", required=True)
    activate_parser.add_argument("--runtime-root", type=Path, required=True)
    activate_parser.add_argument("--approval-token", default="")

    arrival_parser = commands.add_parser("register-arrival")
    arrival_parser.add_argument("campaign_id")
    arrival_parser.add_argument("candidate", type=Path)
    arrival_parser.add_argument("--runtime-root", type=Path, required=True)

    record_parser = commands.add_parser("record-case")
    record_parser.add_argument("campaign_id")
    record_parser.add_argument("result", type=Path)
    record_parser.add_argument("--runtime-root", type=Path, required=True)
    record_parser.add_argument("--store-root", type=Path)

    finalize_parser = commands.add_parser("finalize")
    finalize_parser.add_argument("campaign_id")
    finalize_parser.add_argument("evaluation_metrics", type=Path)
    finalize_parser.add_argument("--observed-through", required=True)
    finalize_parser.add_argument("--finalized-at", required=True)
    finalize_parser.add_argument("--runtime-root", type=Path, required=True)
    finalize_parser.add_argument("--approval-token", default="")

    replay_parser = commands.add_parser("replay")
    replay_parser.add_argument("campaign_id")
    replay_parser.add_argument("--runtime-root", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "preview":
        candidates = _read_json(args.candidates)
        if not isinstance(candidates, list):
            parser.error("preview candidates must be a JSON array")
        result = preview_shadow_campaign(candidates, activated_at=args.activated_at)
    elif args.command == "activate":
        result = activate_shadow_campaign(
            _read_json(args.preview),
            expected_preview_sha256=args.expected_preview_sha256,
            reviewed_at=args.reviewed_at,
            runtime_root=args.runtime_root,
            approval_token=args.approval_token,
        )
    elif args.command == "register-arrival":
        result = register_new_arrival(
            args.campaign_id,
            _read_json(args.candidate),
            runtime_root=args.runtime_root,
        )
    elif args.command == "record-case":
        payload = _read_json(args.result)
        if not isinstance(payload, dict):
            parser.error("record-case result must be a JSON object")
        if args.store_root is None:
            parser.error("--store-root is required to verify the A4 supervisor run")
        supervisor = IdentityEvidenceSupervisor(args.store_root)
        workflow = None
        if payload.get("queue_item") is not None:
            workflow = IdentityReviewWorkflow(args.store_root)
        result = record_shadow_case(
            args.campaign_id,
            payload,
            runtime_root=args.runtime_root,
            review_workflow=workflow,
            evidence_supervisor=supervisor,
        )
    elif args.command == "finalize":
        result = finalize_shadow_campaign(
            args.campaign_id,
            observed_through=args.observed_through,
            finalized_at=args.finalized_at,
            evaluation_metrics=_read_json(args.evaluation_metrics),
            runtime_root=args.runtime_root,
            approval_token=args.approval_token,
        )
    else:
        result = replay_shadow_campaign(
            args.campaign_id,
            runtime_root=args.runtime_root,
        )
    _print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
