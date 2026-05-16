#!/usr/bin/env python3
"""Create operator review artifacts for unresolved transcript filename conflicts."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from cleanup_transcript_filenames import (
    DEFAULT_QUARANTINE_DIR,
    DEFAULT_SERVICE_NAME,
    refresh_store_for_replacements,
    rewrite_artifact_payload,
    rewrite_state_file,
    service_is_active,
    start_service,
    stop_service,
)
from watch_transcriptions import DEFAULT_STATE_PATH

DECISION_OPTIONS = [
    "pending",
    "preserve_both",
    "quarantine_old",
    "keep_target",
    "needs_investigation",
]
CLASSIFICATION_PRIORITY = [
    "distinct_content_preserve_both",
    "partial_overlap_distinct_content",
    "high_overlap_needs_review",
    "metadata_or_format_only_candidate",
    "binary_or_unsupported",
]
DEFAULT_REVIEW_DIR = Path("~/.local/state/transcribe-audio/filename-conflict-reviews")
APPLY_APPROVAL_TOKEN = "APPLY_FILENAME_CONFLICT_REVIEW"


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a human review template and Markdown report for filename cleanup conflicts."
    )
    parser.add_argument(
        "review_export",
        nargs="?",
        type=Path,
        help="JSON produced by cleanup_transcript_filenames.py --export-review.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_REVIEW_DIR,
        help="Directory for generated review artifacts.",
    )
    parser.add_argument("--json-output", type=Path, help="Write the decision template to this path.")
    parser.add_argument("--markdown-output", type=Path, help="Write the Markdown report to this path.")
    parser.add_argument("--apply-review", type=Path, help="Apply decisions from an edited review template JSON.")
    parser.add_argument("--audit-output", type=Path, help="Write the apply-review preview/apply audit JSON to this path.")
    parser.add_argument("--apply", action="store_true", help="Actually apply --apply-review decisions. Default is dry-run.")
    parser.add_argument("--approval-token", help=f"Required with --apply. Use {APPLY_APPROVAL_TOKEN}.")
    parser.add_argument("--quarantine-dir", type=Path, default=DEFAULT_QUARANTINE_DIR)
    parser.add_argument("--manage-service", action="store_true", help="Stop/restart transcribe-watch.service during apply.")
    parser.add_argument("--service-name", default=DEFAULT_SERVICE_NAME)
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--refresh-store", action="store_true")
    parser.add_argument("--store-dir", type=Path, default=Path("~/.transcripts"))
    parser.add_argument("--embedding-provider", default="ollama")
    parser.add_argument("--embedding-model", default="ollama/nomic-embed-text")
    parser.add_argument("--format", choices=["text", "json"], default="text")
    return parser.parse_args(argv)


def load_json(path: Path) -> dict[str, Any]:
    with path.expanduser().open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    tmp_path.replace(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


def item_id(item: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(str(item.get("artifact_path") or "").encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(item.get("clean_base_name") or "").encode("utf-8"))
    for conflict in item.get("target_conflicts", []):
        if not isinstance(conflict, dict):
            continue
        digest.update(b"\0")
        digest.update(str(conflict.get("old_path") or "").encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(conflict.get("target_path") or "").encode("utf-8"))
    return digest.hexdigest()[:16]


def aggregate_classification(item: dict[str, Any]) -> str:
    classifications: list[str] = []
    for conflict in item.get("target_conflicts", []):
        if not isinstance(conflict, dict):
            continue
        diff_summary = conflict.get("diff_summary") if isinstance(conflict.get("diff_summary"), dict) else {}
        classification = str(diff_summary.get("classification") or "").strip()
        if classification:
            classifications.append(classification)
    if not classifications:
        return "unclassified"
    return sorted(
        classifications,
        key=lambda value: CLASSIFICATION_PRIORITY.index(value) if value in CLASSIFICATION_PRIORITY else 99,
    )[0]


def recommended_decision(classification: str) -> str:
    if classification == "distinct_content_preserve_both":
        return "preserve_both"
    if classification in {"partial_overlap_distinct_content", "high_overlap_needs_review"}:
        return "needs_investigation"
    if classification == "metadata_or_format_only_candidate":
        return "quarantine_old"
    return "pending"


def conflict_summary(conflict: dict[str, Any]) -> dict[str, Any]:
    diff_summary = conflict.get("diff_summary") if isinstance(conflict.get("diff_summary"), dict) else {}
    return {
        "role": conflict.get("role"),
        "old_path": conflict.get("old_path"),
        "target_path": conflict.get("target_path"),
        "content_equivalent": conflict.get("content_equivalent"),
        "old_sha256": conflict.get("old_sha256"),
        "target_sha256": conflict.get("target_sha256"),
        "diff_summary": diff_summary,
    }


def build_review_template(review_export: Path, payload: dict[str, Any]) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for source_item in payload.get("items", []):
        if not isinstance(source_item, dict):
            continue
        classification = aggregate_classification(source_item)
        items.append(
            {
                "id": item_id(source_item),
                "decision": "pending",
                "recommended_decision": recommended_decision(classification),
                "allowed_decisions": DECISION_OPTIONS,
                "decision_reason": "",
                "reviewer": "",
                "reviewed_at": "",
                "classification": classification,
                "clean_base_name": source_item.get("clean_base_name"),
                "reason": source_item.get("reason"),
                "event": source_item.get("event"),
                "artifact_path": source_item.get("artifact_path"),
                "source_media_path": source_item.get("source_media_path"),
                "working_media_path": source_item.get("working_media_path"),
                "conflicts": [
                    conflict_summary(conflict)
                    for conflict in source_item.get("target_conflicts", [])
                    if isinstance(conflict, dict)
                ],
                "planned_non_conflicting_operations": source_item.get("operations", []),
            }
        )

    counts = Counter(item["classification"] for item in items)
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_review_export": str(review_export.expanduser().resolve()),
        "status": "pending_operator_review",
        "decision_options": DECISION_OPTIONS,
        "summary": {
            "item_count": len(items),
            "by_classification": dict(counts),
            "by_recommended_decision": dict(Counter(item["recommended_decision"] for item in items)),
        },
        "items": items,
    }


def markdown_report(template: dict[str, Any]) -> str:
    lines = [
        "# Transcript Filename Conflict Review",
        "",
        f"Source review export: `{template['source_review_export']}`",
        f"Created: `{template['created_at']}`",
        "",
        "## Summary",
        "",
        f"- Items: {template['summary']['item_count']}",
    ]
    for classification, count in sorted(template["summary"]["by_classification"].items()):
        lines.append(f"- {classification}: {count}")
    lines.extend(
        [
            "",
            "## Decision Options",
            "",
            "- `preserve_both`: keep both outputs because they represent distinct content or unresolved overlap.",
            "- `quarantine_old`: move the old conflicting output aside and use the canonical target.",
            "- `keep_target`: keep the existing canonical target and leave old paths unresolved for now.",
            "- `needs_investigation`: defer until transcript content is compared manually.",
            "",
            "## Items",
            "",
        ]
    )
    for item in template["items"]:
        lines.extend(
            [
                f"### {item['id']} | {item['clean_base_name']}",
                "",
                f"- Classification: `{item['classification']}`",
                f"- Recommended decision: `{item['recommended_decision']}`",
                f"- Current decision: `{item['decision']}`",
                f"- Reason: `{item['reason']}`",
            ]
        )
        event = item.get("event") if isinstance(item.get("event"), dict) else {}
        if event:
            lines.append(f"- Event: `{event.get('summary')}` `{event.get('start')}`")
        lines.append(f"- Artifact: `{item.get('artifact_path')}`")
        for conflict in item.get("conflicts", []):
            diff = conflict.get("diff_summary") if isinstance(conflict.get("diff_summary"), dict) else {}
            lines.extend(
                [
                    f"- Conflict role: `{conflict.get('role')}`",
                    f"- Body similarity: `{diff.get('body_similarity_ratio', 'n/a')}`; changed spans: `{diff.get('changed_line_spans', 'n/a')}`",
                    f"- Old: `{conflict.get('old_path')}`",
                    f"- Target: `{conflict.get('target_path')}`",
                ]
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def quarantine_path_for(root: Path, source_path: Path) -> Path:
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()[:16] if source_path.exists() else "missing"
    return root / digest[:2] / f"{digest}-{source_path.name}"


def apply_review_item(item: dict[str, Any], *, quarantine_dir: Path, apply: bool) -> dict[str, Any]:
    decision = str(item.get("decision") or "pending")
    result: dict[str, Any] = {
        "id": item.get("id"),
        "decision": decision,
        "clean_base_name": item.get("clean_base_name"),
        "status": "skipped",
        "replacements": {},
        "quarantined": [],
        "moved": [],
        "artifact_path": item.get("artifact_path"),
    }
    if decision in {"pending", "needs_investigation"}:
        result["reason"] = "decision is not actionable"
        return result
    if decision in {"preserve_both", "keep_target"}:
        result["status"] = "recorded_noop"
        result["reason"] = f"decision {decision} does not mutate files"
        return result
    if decision != "quarantine_old":
        result["status"] = "error"
        result["reason"] = f"unsupported decision: {decision}"
        return result

    replacements: dict[str, str] = {}
    quarantine_root = quarantine_dir.expanduser().resolve()
    for conflict in item.get("conflicts", []):
        if not isinstance(conflict, dict):
            continue
        old_path = Path(str(conflict.get("old_path") or "")).expanduser()
        target_path = Path(str(conflict.get("target_path") or "")).expanduser()
        if not old_path.exists():
            result["status"] = "error"
            result["reason"] = f"old conflict path does not exist: {old_path}"
            return result
        if not target_path.exists():
            result["status"] = "error"
            result["reason"] = f"target conflict path does not exist: {target_path}"
            return result
        quarantine_path = quarantine_path_for(quarantine_root, old_path)
        if apply:
            quarantine_path.parent.mkdir(parents=True, exist_ok=True)
            if quarantine_path.exists():
                result["status"] = "error"
                result["reason"] = f"quarantine target already exists: {quarantine_path}"
                return result
            shutil.move(str(old_path), str(quarantine_path))
        replacements[str(old_path.resolve())] = str(target_path.resolve())
        result["quarantined"].append(
            {
                "role": conflict.get("role"),
                "old_path": str(old_path),
                "canonical_path": str(target_path),
                "quarantine_path": str(quarantine_path),
            }
        )

    for operation in item.get("planned_non_conflicting_operations", []):
        if not isinstance(operation, dict):
            continue
        old_path = Path(str(operation.get("old_path") or "")).expanduser()
        new_path = Path(str(operation.get("new_path") or "")).expanduser()
        if not old_path.exists():
            continue
        if new_path.exists():
            result["status"] = "error"
            result["reason"] = f"planned operation target already exists: {new_path}"
            return result
        if apply:
            new_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old_path), str(new_path))
        replacements[str(old_path.resolve())] = str(new_path.resolve())
        result["moved"].append({"role": operation.get("role"), "old_path": str(old_path), "new_path": str(new_path)})

    artifact_path = Path(str(item.get("artifact_path") or "")).expanduser()
    final_artifact = Path(replacements.get(str(artifact_path.resolve()), str(artifact_path.resolve())))
    if apply and final_artifact.exists():
        rewrite_artifact_payload(final_artifact, replacements)
    result["artifact_path"] = str(final_artifact)
    result["replacements"] = replacements
    result["status"] = "applied" if apply else "planned"
    result["dry_run"] = not apply
    return result


def apply_review_template(args: argparse.Namespace) -> dict[str, Any]:
    if args.apply and args.approval_token != APPLY_APPROVAL_TOKEN:
        raise ValueError(f"--apply requires --approval-token {APPLY_APPROVAL_TOKEN}")
    if args.apply and service_is_active(args.service_name) and not args.manage_service:
        raise RuntimeError(f"{args.service_name} is active; use --manage-service for apply")

    template = load_json(args.apply_review)
    results: list[dict[str, Any]] = []
    service_was_active = False
    try:
        if args.apply and args.manage_service:
            service_was_active = stop_service(args.service_name)
        for item in template.get("items", []):
            if isinstance(item, dict):
                results.append(apply_review_item(item, quarantine_dir=args.quarantine_dir, apply=args.apply))
        replacements: dict[str, str] = {}
        artifact_paths: list[str] = []
        for result in results:
            if result.get("status") in {"planned", "applied"}:
                replacements.update(result.get("replacements", {}))
                if result.get("artifact_path"):
                    artifact_paths.append(str(result["artifact_path"]))
        state_updated = (
            rewrite_state_file(args.state_file.expanduser().resolve(), replacements)
            if args.apply and replacements
            else False
        )
        store_refreshed = (
            refresh_store_for_replacements(
                replacements,
                artifact_paths,
                store_dir=args.store_dir,
                embedding_provider=args.embedding_provider,
                embedding_model=args.embedding_model,
            )
            if args.apply and args.refresh_store and replacements
            else []
        )
    finally:
        if args.apply and args.manage_service and service_was_active:
            start_service(args.service_name)

    counts = Counter(str(result.get("status")) for result in results)
    result = {
        "mode": "apply" if args.apply else "preview",
        "review_path": str(args.apply_review.expanduser().resolve()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "items": len(results),
            "by_status": dict(counts),
            "mutating_decisions": sum(1 for result in results if result.get("status") in {"planned", "applied"}),
        },
        "state_updated": state_updated,
        "store_refreshed": store_refreshed,
        "results": results,
    }
    if args.audit_output:
        write_json(args.audit_output.expanduser().resolve(), result)
        result["audit_path"] = str(args.audit_output.expanduser().resolve())
    return result


def default_output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    output_dir = args.output_dir.expanduser().resolve()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = args.json_output.expanduser().resolve() if args.json_output else output_dir / f"filename-conflict-review-{stamp}.json"
    markdown_path = (
        args.markdown_output.expanduser().resolve()
        if args.markdown_output
        else output_dir / f"filename-conflict-review-{stamp}.md"
    )
    return json_path, markdown_path


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    if args.apply_review:
        result = apply_review_template(args)
        if args.format == "json":
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print(f"Mode: {result['mode']}")
            print(f"Review: {result['review_path']}")
            print(f"Items: {result['summary']['items']}")
            print(f"Statuses: {result['summary']['by_status']}")
        return 0
    if args.apply:
        print("Error: --apply requires --apply-review", file=sys.stderr)
        return 2
    if not args.review_export:
        print("Error: review_export is required unless --apply-review is used", file=sys.stderr)
        return 2

    review_export = args.review_export.expanduser().resolve()
    payload = load_json(review_export)
    template = build_review_template(review_export, payload)
    json_path, markdown_path = default_output_paths(args)
    write_json(json_path, template)
    write_text(markdown_path, markdown_report(template))
    result = {
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
        "summary": template["summary"],
    }
    if args.format == "json":
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"Review JSON: {json_path}")
        print(f"Review Markdown: {markdown_path}")
        print(f"Items: {template['summary']['item_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
