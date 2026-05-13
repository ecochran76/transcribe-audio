#!/usr/bin/env python3
"""
Apply reviewed Graphiti memory-harvest candidates from a deposition preview.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Optional

from deposition_artifacts import (
    AppliedMemoryHarvestCandidate,
    MemoryHarvestApplyResult,
    normalize_string,
)
from transcribe_common import TranscriptionError

MEMORY_HARVEST_APPLY_JSON_STDOUT_PREFIX = "MEMORY_HARVEST_APPLY_JSON="
MEMORY_HARVEST_REVIEW_JSON_STDOUT_PREFIX = "MEMORY_HARVEST_REVIEW_JSON="
APPROVAL_TOKEN = "APPROVE_GRAPHITI_MEMORY_HARVEST"
REVIEW_DECISIONS = {"approved", "rejected", "pending"}


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview or apply reviewed Graphiti memory-harvest candidates from a deposit preview."
    )
    parser.add_argument("preview", type=Path, help="Path to a *.deposit-preview.json artifact.")
    parser.add_argument("--output-dir", type=Path, help="Directory for apply result output. Defaults beside preview.")
    parser.add_argument(
        "--init-review",
        action="store_true",
        help="Write a *.memory-harvest-review.json template with pending candidate decisions.",
    )
    parser.add_argument(
        "--review-file",
        type=Path,
        help="Path to a memory harvest review JSON file. Only approved candidates are applied.",
    )
    parser.add_argument(
        "--review-output",
        type=Path,
        help="Path for --init-review output. Defaults beside preview or under --output-dir.",
    )
    parser.add_argument("--apply", action="store_true", help="Actually write approved candidates to Graphiti.")
    parser.add_argument(
        "--approval-token",
        help=f"Required with --apply. Must be {APPROVAL_TOKEN}.",
    )
    parser.add_argument(
        "--allow-review-required",
        action="store_true",
        help="Allow apply even when the preview route required review.",
    )
    parser.add_argument(
        "--allow-warnings",
        action="store_true",
        help="Allow apply when the preview carries warnings.",
    )
    parser.add_argument(
        "--candidate-id",
        action="append",
        dest="candidate_ids",
        help="Apply only this candidate id. Repeatable. Defaults to all candidates.",
    )
    parser.add_argument(
        "--skip-duplicate-check",
        action="store_true",
        help="Skip Graphiti duplicate preflight during --apply.",
    )
    parser.add_argument(
        "--graphiti-command",
        default=str(Path("~/.local/bin/graphiti-runtime").expanduser()),
        help="Path to graphiti-runtime.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=180.0, help="Graphiti write timeout per candidate.")
    return parser.parse_args(argv)


def load_preview(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    except OSError as exc:
        raise TranscriptionError(f"Failed to read deposition preview {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise TranscriptionError(f"Deposition preview {path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise TranscriptionError(f"Deposition preview {path} must contain a JSON object.")
    return payload


def output_path(preview_path: Path, output_dir: Optional[Path]) -> Path:
    base_name = preview_path.name
    if base_name.endswith(".deposit-preview.json"):
        base_name = base_name[: -len(".deposit-preview.json")]
    else:
        base_name = preview_path.stem
    directory = output_dir.expanduser() if output_dir else preview_path.parent
    return directory / f"{base_name}.memory-harvest-apply.json"


def review_output_path(preview_path: Path, output_dir: Optional[Path], review_output: Optional[Path]) -> Path:
    if review_output:
        return review_output.expanduser()
    base_name = preview_path.name
    if base_name.endswith(".deposit-preview.json"):
        base_name = base_name[: -len(".deposit-preview.json")]
    else:
        base_name = preview_path.stem
    directory = output_dir.expanduser() if output_dir else preview_path.parent
    return directory / f"{base_name}.memory-harvest-review.json"


def candidate_id(candidate: dict[str, Any]) -> str:
    return normalize_string(candidate.get("candidate_id"))


def select_candidates(preview: dict[str, Any], ids: Optional[list[str]]) -> list[dict[str, Any]]:
    candidates = [item for item in preview.get("memory_candidates") or [] if isinstance(item, dict)]
    if not ids:
        return candidates
    wanted = {normalize_string(item) for item in ids if normalize_string(item)}
    selected = [candidate for candidate in candidates if candidate_id(candidate) in wanted]
    missing = sorted(wanted - {candidate_id(candidate) for candidate in selected})
    if missing:
        raise TranscriptionError(f"Requested memory candidate id(s) not found: {', '.join(missing)}")
    return selected


def validate_apply_allowed(args: argparse.Namespace, preview: dict[str, Any]) -> None:
    if preview.get("review_required") and not args.allow_review_required:
        raise TranscriptionError(
            "Refusing memory harvest because the deposition preview requires review. "
            "Pass --allow-review-required only after review."
        )
    warnings = [normalize_string(item) for item in preview.get("warnings") or [] if normalize_string(item)]
    if warnings and not args.allow_warnings:
        raise TranscriptionError(
            "Refusing memory harvest because the deposition preview carries warnings. "
            "Pass --allow-warnings only after review."
        )
    if args.apply and args.approval_token != APPROVAL_TOKEN:
        raise TranscriptionError(f"--apply requires --approval-token {APPROVAL_TOKEN}.")


def review_template(preview_path: Path, preview: dict[str, Any]) -> dict[str, Any]:
    candidates = [item for item in preview.get("memory_candidates") or [] if isinstance(item, dict)]
    return {
        "schema_version": 1,
        "source": "transcribe-audio.memory_harvest_review.v1",
        "source_preview_path": str(preview_path),
        "instructions": (
            "Set each decision to approved, rejected, or pending. "
            "memory_harvest_apply.py --review-file applies only approved candidates."
        ),
        "candidates": [
            {
                "candidate_id": candidate_id(candidate),
                "decision": "pending",
                "reason": "",
                "kind": normalize_string(candidate.get("kind")) or "memory",
                "target_group_id": normalize_string(candidate.get("target_group_id")) or "transcribe_audio_main",
                "text": normalize_string(candidate.get("text")),
                "evidence": normalize_string(candidate.get("evidence")),
            }
            for candidate in candidates
        ],
    }


def write_review_template(args: argparse.Namespace, preview_path: Path, preview: dict[str, Any]) -> Path:
    path = review_output_path(preview_path, args.output_dir, args.review_output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(review_template(preview_path, preview), handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    return path


def load_review(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    except OSError as exc:
        raise TranscriptionError(f"Failed to read memory harvest review {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise TranscriptionError(f"Memory harvest review {path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise TranscriptionError(f"Memory harvest review {path} must contain a JSON object.")
    return payload


def review_decisions(review: Optional[dict[str, Any]]) -> dict[str, dict[str, str]]:
    if not review:
        return {}
    decisions: dict[str, dict[str, str]] = {}
    for item in review.get("candidates") or []:
        if not isinstance(item, dict):
            continue
        item_id = normalize_string(item.get("candidate_id"))
        if not item_id:
            continue
        decision = normalize_string(item.get("decision")).lower() or "pending"
        if decision not in REVIEW_DECISIONS:
            raise TranscriptionError(
                f"Review decision for candidate {item_id} must be one of: {', '.join(sorted(REVIEW_DECISIONS))}."
            )
        decisions[item_id] = {
            "decision": decision,
            "reason": normalize_string(item.get("reason")),
        }
    return decisions


def review_for_candidate(candidate: dict[str, Any], decisions: dict[str, dict[str, str]]) -> dict[str, str]:
    if not decisions:
        return {"decision": "unreviewed", "reason": ""}
    item_id = candidate_id(candidate)
    return decisions.get(item_id, {"decision": "missing", "reason": "No review decision found for candidate."})


def memory_body(candidate: dict[str, Any], preview: dict[str, Any]) -> dict[str, Any]:
    selected = preview.get("selected_candidate") if isinstance(preview.get("selected_candidate"), dict) else {}
    return {
        "schema_version": 1,
        "source": "transcribe-audio.memory_harvest.v1",
        "candidate_id": candidate_id(candidate),
        "kind": normalize_string(candidate.get("kind")) or "memory",
        "text": normalize_string(candidate.get("text")),
        "evidence": normalize_string(candidate.get("evidence")),
        "target_group_id": normalize_string(candidate.get("target_group_id")) or "transcribe_audio_main",
        "source_readout_path": normalize_string(candidate.get("source_readout_path")),
        "source_ids": [normalize_string(item) for item in candidate.get("source_ids") or [] if normalize_string(item)],
        "selected_candidate": {
            "label": normalize_string(selected.get("label")),
            "target_kind": normalize_string(selected.get("target_kind")),
            "target_id": normalize_string(selected.get("target_id")),
            "confidence": selected.get("confidence"),
        },
        "preview_warnings": [normalize_string(item) for item in preview.get("warnings") or [] if normalize_string(item)],
    }


def write_body_file(body: dict[str, Any]) -> Path:
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        prefix="transcribe-memory-harvest-",
        suffix=".json",
        delete=False,
        encoding="utf-8",
    )
    with handle:
        json.dump(body, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    return Path(handle.name)


def graphiti_command(args: argparse.Namespace, candidate: dict[str, Any], body_file: Path) -> list[str]:
    group_id = normalize_string(candidate.get("target_group_id")) or "transcribe_audio_main"
    name = f"transcribe-audio memory harvest {candidate_id(candidate)}"
    query = normalize_string(candidate.get("text"))[:240] or candidate_id(candidate)
    return [
        args.graphiti_command,
        "benchmark-write",
        "--group-id",
        group_id,
        "--name",
        name,
        "--source-description",
        f"transcribe-audio memory harvest candidate {candidate_id(candidate)}",
        "--body-file",
        str(body_file),
        "--query",
        query,
        "--timeout-seconds",
        str(args.timeout_seconds),
    ]


def graphiti_discover_command(args: argparse.Namespace, candidate: dict[str, Any]) -> list[str]:
    group_id = normalize_string(candidate.get("target_group_id")) or "transcribe_audio_main"
    query = normalize_string(candidate.get("text"))[:240] or candidate_id(candidate)
    return [
        args.graphiti_command,
        "discover",
        "--group-id",
        group_id,
        query,
    ]


def compact_duplicate_check(
    candidate: dict[str, Any],
    command: list[str],
    payload: dict[str, Any],
    returncode: int,
) -> dict[str, Any]:
    item_id = candidate_id(candidate)
    episodes = payload.get("episodes") if isinstance(payload.get("episodes"), list) else []
    exact_matches = []
    for episode in episodes:
        if not isinstance(episode, dict):
            continue
        source_description = normalize_string(episode.get("source_description"))
        content_preview = normalize_string(episode.get("content_preview"))
        if item_id and (item_id in source_description or f'"candidate_id": "{item_id}"' in content_preview):
            exact_matches.append(
                {
                    "uuid": normalize_string(episode.get("uuid")),
                    "name": normalize_string(episode.get("name")),
                    "source_description": source_description,
                }
            )
    return {
        "command": command,
        "returncode": returncode,
        "status": payload.get("status") if isinstance(payload, dict) else {},
        "episode_count": payload.get("episode_count") if isinstance(payload, dict) else None,
        "fact_count": payload.get("fact_count") if isinstance(payload, dict) else None,
        "node_count": payload.get("node_count") if isinstance(payload, dict) else None,
        "exact_duplicate": bool(exact_matches),
        "exact_matches": exact_matches,
    }


def duplicate_check(
    args: argparse.Namespace,
    candidate: dict[str, Any],
    *,
    runner=subprocess.run,
) -> dict[str, Any]:
    command = graphiti_discover_command(args, candidate)
    try:
        completed = runner(command, text=True, capture_output=True, timeout=min(args.timeout_seconds, 60), check=False)
    except FileNotFoundError as exc:
        raise TranscriptionError(f"Graphiti duplicate check requires `{args.graphiti_command}`.") from exc
    except subprocess.TimeoutExpired as exc:
        raise TranscriptionError("Graphiti duplicate check timed out.") from exc
    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError:
        payload = {"error": "Graphiti duplicate check did not return JSON."}
    if not isinstance(payload, dict):
        payload = {"error": "Graphiti duplicate check returned non-object JSON."}
    return compact_duplicate_check(candidate, command, payload, completed.returncode)


def compact_graphiti_result(payload: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "command",
        "group_id",
        "status",
        "job_id",
        "episode_uuid",
        "verified",
        "duration_seconds",
        "error",
    )
    return {key: payload.get(key) for key in keys if key in payload}


def apply_candidate(
    args: argparse.Namespace,
    candidate: dict[str, Any],
    preview: dict[str, Any],
    *,
    review: Optional[dict[str, str]] = None,
    runner=subprocess.run,
) -> AppliedMemoryHarvestCandidate:
    review = review or {"decision": "unreviewed", "reason": ""}
    decision = review.get("decision", "unreviewed")
    review_reason = review.get("reason", "")
    if decision == "rejected":
        return applied_candidate(
            candidate,
            status="rejected",
            reason=review_reason or "Candidate rejected during operator review.",
            review=review,
        )
    if decision in {"pending", "missing"}:
        return applied_candidate(
            candidate,
            status="skipped",
            reason=review_reason or "Candidate is not approved for apply.",
            review=review,
        )
    body = memory_body(candidate, preview)
    if not body["text"]:
        return applied_candidate(candidate, status="skipped", reason="Candidate text is empty.", review=review)
    if not args.apply:
        command = graphiti_command(args, candidate, Path("<BODY_FILE_CREATED_ON_APPLY>"))
        duplicate_command = [] if args.skip_duplicate_check else graphiti_discover_command(args, candidate)
        duplicate = {"status": "planned", "command": duplicate_command} if duplicate_command else {"status": "skipped"}
        return applied_candidate(
            candidate,
            status="planned",
            reason="Dry run; no Graphiti write attempted.",
            command=command,
            review=review,
            duplicate_check=duplicate,
        )
    duplicate = {}
    if not args.skip_duplicate_check:
        duplicate = duplicate_check(args, candidate, runner=runner)
        if duplicate.get("returncode") != 0:
            return applied_candidate(
                candidate,
                status="duplicate_check_failed",
                reason="Graphiti duplicate check failed; write was not attempted.",
                review=review,
                duplicate_check=duplicate,
            )
        if duplicate.get("exact_duplicate"):
            return applied_candidate(
                candidate,
                status="duplicate_skipped",
                reason="Exact candidate already exists in Graphiti.",
                review=review,
                duplicate_check=duplicate,
            )
    body_file = write_body_file(body)
    command = graphiti_command(args, candidate, body_file)
    try:
        completed = runner(command, text=True, capture_output=True, timeout=args.timeout_seconds + 30, check=False)
    except FileNotFoundError as exc:
        raise TranscriptionError(f"Graphiti memory harvest requires `{args.graphiti_command}`.") from exc
    except subprocess.TimeoutExpired as exc:
        raise TranscriptionError(f"Graphiti memory harvest timed out after {args.timeout_seconds:g} seconds.") from exc
    finally:
        try:
            body_file.unlink()
        except OSError:
            pass
    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError:
        payload = {"error": "Graphiti command did not return JSON."}
    status = "applied" if completed.returncode == 0 else "failed"
    return applied_candidate(
        candidate,
        status=status,
        reason="" if completed.returncode == 0 else "Graphiti write command failed.",
        command=command,
        review=review,
        duplicate_check=duplicate,
        graphiti_result={
            **compact_graphiti_result(payload if isinstance(payload, dict) else {}),
            "returncode": completed.returncode,
        },
    )


def applied_candidate(
    candidate: dict[str, Any],
    *,
    status: str,
    reason: str = "",
    command: Optional[list[str]] = None,
    review: Optional[dict[str, str]] = None,
    duplicate_check: Optional[dict[str, Any]] = None,
    graphiti_result: Optional[dict[str, Any]] = None,
) -> AppliedMemoryHarvestCandidate:
    review = review or {"decision": "unreviewed", "reason": ""}
    return AppliedMemoryHarvestCandidate(
        candidate_id=candidate_id(candidate),
        target_group_id=normalize_string(candidate.get("target_group_id")) or "transcribe_audio_main",
        status=status,
        kind=normalize_string(candidate.get("kind")) or "memory",
        reason=reason,
        review_decision=review.get("decision", ""),
        review_reason=review.get("reason", ""),
        source_readout_path=normalize_string(candidate.get("source_readout_path")),
        source_ids=[normalize_string(item) for item in candidate.get("source_ids") or [] if normalize_string(item)],
        duplicate_check=duplicate_check or {},
        graphiti_command=command or [],
        graphiti_result=graphiti_result or {},
    )


def apply_preview(args: argparse.Namespace, *, runner=subprocess.run) -> Path:
    preview_path = args.preview.expanduser().resolve()
    preview = load_preview(preview_path)
    if args.init_review:
        return write_review_template(args, preview_path, preview)
    validate_apply_allowed(args, preview)
    candidates = select_candidates(preview, args.candidate_ids)
    review = load_review(args.review_file) if args.review_file else None
    decisions = review_decisions(review)
    applied = [
        apply_candidate(args, candidate, preview, review=review_for_candidate(candidate, decisions), runner=runner)
        for candidate in candidates
    ]
    warnings = [normalize_string(item) for item in preview.get("warnings") or [] if normalize_string(item)]
    result = MemoryHarvestApplyResult(
        source_preview_path=str(preview_path),
        mode="apply" if args.apply else "preview",
        candidates=applied,
        warnings=warnings,
    )
    path = output_path(preview_path, args.output_dir)
    result.write_json(path)
    return path


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    try:
        path = apply_preview(args)
    except TranscriptionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    if args.init_review:
        print(f"Writing memory harvest review JSON to {path}...")
        print(f"{MEMORY_HARVEST_REVIEW_JSON_STDOUT_PREFIX}{path}")
    else:
        print(f"Writing memory harvest apply JSON to {path}...")
        print(f"{MEMORY_HARVEST_APPLY_JSON_STDOUT_PREFIX}{path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
