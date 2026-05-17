#!/usr/bin/env python3
"""
Check first-pass readout artifacts before scaling provider batches.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Optional


REQUIRED_LIST_FIELDS = (
    "participants",
    "topics",
    "action_items",
    "matter_candidates",
    "memory_candidates",
)
QUALITY_JSON_STDOUT_PREFIX = "READOUT_QUALITY_JSON="


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate generated first-pass readout artifacts.")
    parser.add_argument("paths", nargs="*", type=Path, help="Readout JSON files to check.")
    parser.add_argument("--manifest", type=Path, help="Batch manifest with materialized readout paths.")
    parser.add_argument("--min-summary-chars", type=int, default=200)
    parser.add_argument("--format", choices=("json", "text"), default="json")
    return parser.parse_args(argv)


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return payload


def readout_paths_from_manifest(path: Path) -> list[Path]:
    payload = load_json(path.expanduser())
    paths: list[Path] = []
    for item in payload.get("materialized") or []:
        if isinstance(item, dict) and item.get("readout_json"):
            paths.append(Path(str(item["readout_json"])).expanduser())
    return paths


def add_finding(findings: list[dict[str, str]], severity: str, code: str, message: str) -> None:
    findings.append({"severity": severity, "code": code, "message": message})


def check_readout(path: Path, *, min_summary_chars: int) -> dict[str, Any]:
    resolved = path.expanduser()
    findings: list[dict[str, str]] = []
    metrics: dict[str, Any] = {}
    try:
        payload = load_json(resolved)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return {
            "path": str(resolved),
            "status": "fail",
            "findings": [{"severity": "fail", "code": "invalid_json", "message": str(exc)}],
            "metrics": metrics,
        }

    if payload.get("schema_version") != 1:
        add_finding(findings, "fail", "schema_version", "Expected schema_version=1.")
    for key in ("title", "summary", "source_artifact_path", "generated_at"):
        if not str(payload.get(key) or "").strip():
            add_finding(findings, "fail", f"missing_{key}", f"Missing non-empty {key}.")

    summary = str(payload.get("summary") or "")
    metrics["summary_chars"] = len(summary)
    if len(summary.strip()) < min_summary_chars:
        add_finding(findings, "warn", "short_summary", f"Summary is shorter than {min_summary_chars} characters.")

    source_path = Path(str(payload.get("source_artifact_path") or "")).expanduser()
    metrics["source_exists"] = bool(str(source_path)) and source_path.exists()
    if str(source_path) and not source_path.exists():
        add_finding(findings, "fail", "missing_source_artifact", "source_artifact_path does not exist.")

    markdown_path = resolved.with_suffix(".md")
    metrics["markdown_exists"] = markdown_path.exists()
    if not markdown_path.exists():
        add_finding(findings, "fail", "missing_markdown", "Matching Markdown readout is missing.")
    elif markdown_path.stat().st_size <= 0:
        add_finding(findings, "fail", "empty_markdown", "Matching Markdown readout is empty.")

    for key in REQUIRED_LIST_FIELDS:
        value = payload.get(key)
        count = len(value) if isinstance(value, list) else 0
        metrics[f"{key}_count"] = count
        if not isinstance(value, list):
            add_finding(findings, "fail", f"{key}_not_list", f"{key} must be a list.")
        elif count == 0:
            add_finding(findings, "warn", f"empty_{key}", f"{key} is empty.")

    status = "fail" if any(item["severity"] == "fail" for item in findings) else "warn" if findings else "pass"
    return {
        "path": str(resolved),
        "status": status,
        "title": str(payload.get("title") or ""),
        "findings": findings,
        "metrics": metrics,
    }


def quality_report(paths: list[Path], *, min_summary_chars: int) -> dict[str, Any]:
    checks = [check_readout(path, min_summary_chars=min_summary_chars) for path in paths]
    status = "fail" if any(item["status"] == "fail" for item in checks) else "warn" if any(
        item["status"] == "warn" for item in checks
    ) else "pass"
    return {
        "status": status,
        "checked_count": len(checks),
        "pass_count": sum(1 for item in checks if item["status"] == "pass"),
        "warn_count": sum(1 for item in checks if item["status"] == "warn"),
        "fail_count": sum(1 for item in checks if item["status"] == "fail"),
        "checks": checks,
    }


def format_text(report: dict[str, Any]) -> str:
    lines = [
        f"Readout quality: {report['status']} ({report['pass_count']} pass, {report['warn_count']} warn, {report['fail_count']} fail)"
    ]
    for check in report["checks"]:
        lines.append(f"- {check['status']}: {check['path']}")
        for finding in check["findings"]:
            lines.append(f"  {finding['severity']}: {finding['code']} - {finding['message']}")
    return "\n".join(lines)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    paths = [path.expanduser() for path in args.paths]
    if args.manifest:
        paths.extend(readout_paths_from_manifest(args.manifest))
    report = quality_report(paths, min_summary_chars=args.min_summary_chars)
    if args.format == "text":
        print(format_text(report))
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
        print(f"{QUALITY_JSON_STDOUT_PREFIX}{report['status']}")
    return 1 if report["status"] == "fail" else 0


if __name__ == "__main__":
    sys.exit(main())
