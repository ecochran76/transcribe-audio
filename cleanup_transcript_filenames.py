#!/usr/bin/env python3
"""
Plan and apply cleanup for transcript artifacts created with duplicated calendar
prefixes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

from docx import Document

from transcribe_common import build_event_base_name
from watch_transcriptions import AUDIO_VIDEO_EXTENSIONS, DEFAULT_STATE_PATH, dump_json, load_json

DEFAULT_SERVICE_NAME = "transcribe-watch.service"
DEFAULT_QUARANTINE_DIR = Path("~/.local/state/transcribe-audio/filename-cleanup-quarantine")
DATE_PREFIX_RE = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}-\d{2} ")
OTHER_EVENTS_RE = re.compile(r"\band \d+ other\(s\)\b", re.IGNORECASE)
TRANSCRIPT_SUFFIXES = {
    "artifact": " Transcript.transcript.json",
    "docx": " Transcript.docx",
    "txt": " Transcript.txt",
    "srt": " Transcript.srt",
}


class CleanupError(RuntimeError):
    """Raised for cleanup planning or apply failures."""


@dataclass
class RenameOperation:
    old_path: Path
    new_path: Path
    role: str

    def to_dict(self) -> dict[str, str]:
        return {
            "role": self.role,
            "old_path": str(self.old_path),
            "new_path": str(self.new_path),
        }


@dataclass
class CleanupPlan:
    artifact_path: Path
    clean_base_name: str
    operations: list[RenameOperation] = field(default_factory=list)
    replacements: dict[str, str] = field(default_factory=dict)
    skipped: bool = False
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_path": str(self.artifact_path),
            "clean_base_name": self.clean_base_name,
            "operations": [operation.to_dict() for operation in self.operations],
            "replacements": self.replacements,
            "skipped": self.skipped,
            "reason": self.reason,
        }


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clean duplicated calendar prefixes from transcript media/output names. "
            "Dry-run is the default."
        )
    )
    parser.add_argument("roots", nargs="+", type=Path, help="Files or directories containing *.transcript.json artifacts.")
    parser.add_argument("--recursive", action="store_true", help="Scan directories recursively.")
    parser.add_argument("--apply", action="store_true", help="Apply planned renames and pointer rewrites.")
    parser.add_argument(
        "--manage-service",
        action="store_true",
        help="Stop transcribe-watch.service before apply and restart it afterwards.",
    )
    parser.add_argument("--service-name", default=DEFAULT_SERVICE_NAME)
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument(
        "--refresh-store",
        action="store_true",
        help="After apply, delete stale store rows for renamed artifacts and re-ingest updated sidecars.",
    )
    parser.add_argument("--store-dir", type=Path, default=Path("~/.transcripts"))
    parser.add_argument("--embedding-provider", default="ollama")
    parser.add_argument("--embedding-model", default="ollama/nomic-embed-text")
    parser.add_argument("--limit", type=int, help="Limit number of actionable plans.")
    parser.add_argument(
        "--export-review",
        type=Path,
        help="Write a compact JSON review file for skipped/conflicting plans.",
    )
    parser.add_argument(
        "--resolve-identical-conflicts",
        action="store_true",
        help="In apply mode, quarantine old conflict files whose canonical targets have identical content.",
    )
    parser.add_argument("--quarantine-dir", type=Path, default=DEFAULT_QUARANTINE_DIR)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON output.")
    return parser.parse_args(argv)


def load_json_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise CleanupError(f"{path} is not a JSON object.")
    return payload


def write_json_file(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    tmp_path.replace(path)


def iter_artifact_paths(roots: Iterable[Path], *, recursive: bool) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        expanded = root.expanduser()
        if expanded.is_file():
            candidates = [expanded]
        elif expanded.is_dir():
            iterator = expanded.rglob("*.transcript.json") if recursive else expanded.glob("*.transcript.json")
            candidates = list(iterator)
        else:
            continue
        for candidate in candidates:
            if not candidate.name.endswith(".transcript.json"):
                continue
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            paths.append(resolved)
    return sorted(paths, key=lambda item: str(item).lower())


def parse_event_start(payload: dict[str, Any]) -> Optional[datetime]:
    event = payload.get("event")
    if not isinstance(event, dict):
        return None
    raw_start = str(event.get("start") or "").strip()
    if not raw_start:
        return None
    try:
        return datetime.fromisoformat(raw_start.replace("Z", "+00:00"))
    except ValueError:
        return None


def clean_base_name(payload: dict[str, Any]) -> Optional[str]:
    event = payload.get("event")
    if not isinstance(event, dict):
        return None
    event_start = parse_event_start(payload)
    if event_start is None:
        return None
    event_summary = str(event.get("summary") or "Untitled Event")
    source_path = Path(str(payload.get("source_media_path") or payload.get("working_media_path") or "recording"))
    return build_event_base_name(event_start, event_summary, source_path.stem)


def output_target_for(key: str, old_path: Path, clean_base: str) -> Path:
    suffix = TRANSCRIPT_SUFFIXES.get(key)
    if suffix:
        return old_path.with_name(f"{clean_base}{suffix}")
    return old_path.with_name(f"{clean_base}{old_path.suffix}")


def existing_media_paths(payload: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    for key in ("working_media_path", "source_media_path"):
        raw_path = str(payload.get(key) or "").strip()
        if not raw_path:
            continue
        path = Path(raw_path).expanduser()
        if path.suffix.lower() not in AUDIO_VIDEO_EXTENSIONS:
            continue
        if path.exists() and path.is_file() and path not in paths:
            paths.append(path.resolve())
    return paths


def add_rename(plan: CleanupPlan, old_path: Path, new_path: Path, role: str) -> None:
    old_resolved = old_path.expanduser().resolve()
    new_resolved = new_path.expanduser().resolve()
    if old_resolved == new_resolved:
        return
    if not old_resolved.exists():
        return
    if new_resolved.exists():
        plan.skipped = True
        plan.reason = f"target already exists: {new_resolved}"
        return
    plan.operations.append(RenameOperation(old_resolved, new_resolved, role))
    plan.replacements[str(old_resolved)] = str(new_resolved)


def add_pointer_replacement(plan: CleanupPlan, old_path: Path, new_path: Path) -> None:
    old_resolved = old_path.expanduser().resolve()
    new_resolved = new_path.expanduser().resolve()
    if old_resolved != new_resolved:
        plan.replacements[str(old_resolved)] = str(new_resolved)


def media_name_needs_cleanup(path: Path) -> bool:
    stem = path.stem
    return (
        len(DATE_PREFIX_RE.findall(stem)) >= 2
        or stem.endswith(" (1)")
        or OTHER_EVENTS_RE.search(stem) is not None
    )


def normalized_text(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def comparable_content(path: Path, role: str) -> str:
    if role == "output:artifact":
        payload = load_json_file(path)
        return normalized_text(str(payload.get("transcript_text") or ""))
    if role == "output:txt":
        return normalized_text(path.read_text(encoding="utf-8", errors="replace"))
    if role == "output:docx":
        doc = Document(str(path))
        return normalized_text("\n".join(paragraph.text for paragraph in doc.paragraphs))
    return sha256_file(path)


def content_equivalent(old_path: Path, new_path: Path, role: str) -> bool:
    if not old_path.exists() or not new_path.exists() or old_path.resolve() == new_path.resolve():
        return False
    if role in {"output:artifact", "output:txt", "output:docx"}:
        return comparable_content(old_path, role) == comparable_content(new_path, role)
    return sha256_file(old_path) == sha256_file(new_path)


def event_review(payload: dict[str, Any]) -> dict[str, Any]:
    event = payload.get("event") if isinstance(payload.get("event"), dict) else {}
    matching_calendars = event.get("matching_calendars") if isinstance(event.get("matching_calendars"), list) else []
    return {
        "summary": event.get("summary"),
        "start": event.get("start"),
        "end": event.get("end"),
        "participants": event.get("participants") if isinstance(event.get("participants"), list) else [],
        "matching_calendar_count": len(matching_calendars),
        "matching_calendars": [
            {
                "calendar_id": item.get("calendar_id"),
                "calendar_summary": item.get("calendar_summary"),
                "event_summary": item.get("event_summary"),
                "coverage": item.get("coverage"),
            }
            for item in matching_calendars
            if isinstance(item, dict)
        ],
    }


def target_conflicts(plan: CleanupPlan) -> list[dict[str, str]]:
    conflicts: list[dict[str, str]] = []
    for operation in plan.operations:
        if operation.new_path.exists():
            conflicts.append(
                {
                    "role": operation.role,
                    "old_path": str(operation.old_path),
                    "target_path": str(operation.new_path),
                }
            )
    try:
        payload = load_json_file(plan.artifact_path)
    except (OSError, json.JSONDecodeError, CleanupError):
        return conflicts
    output_paths = payload.get("output_paths") if isinstance(payload.get("output_paths"), dict) else {}
    for key, raw_path in output_paths.items():
        old_path = Path(str(raw_path)).expanduser()
        target_path = output_target_for(str(key), old_path, plan.clean_base_name)
        if old_path.exists() and target_path.exists() and old_path.resolve() != target_path.resolve():
            conflicts.append(
                {
                    "role": f"output:{key}",
                    "old_path": str(old_path.resolve()),
                    "target_path": str(target_path.resolve()),
                }
            )
    return conflicts


def suggested_review_action(plan: CleanupPlan) -> str:
    if plan.reason == "missing usable event metadata":
        return "inspect sidecar/event metadata before renaming"
    if plan.reason == "already canonical":
        return "no action needed"
    if plan.reason == "shared media has conflicting canonical targets":
        return "choose the canonical media title for shared overlapping calendar artifacts"
    if plan.reason == "canonical media target still contains cleanup noise":
        return "choose one event title or keep shared media unchanged"
    if plan.reason.startswith("target already exists:"):
        return "compare duplicate artifact/media contents before merge or deletion"
    return "manual review required"


def plan_artifact_cleanup(artifact_path: Path) -> CleanupPlan:
    payload = load_json_file(artifact_path)
    clean_base = clean_base_name(payload)
    if not clean_base:
        return CleanupPlan(artifact_path, "", skipped=True, reason="missing usable event metadata")

    plan = CleanupPlan(artifact_path=artifact_path, clean_base_name=clean_base)
    output_paths = payload.get("output_paths") if isinstance(payload.get("output_paths"), dict) else {}
    for key, raw_path in output_paths.items():
        if not raw_path:
            continue
        old_path = Path(str(raw_path)).expanduser()
        if old_path.exists() and old_path.is_file():
            add_rename(plan, old_path, output_target_for(str(key), old_path, clean_base), f"output:{key}")

    artifact_resolved = artifact_path.expanduser().resolve()
    if str(artifact_resolved) not in plan.replacements:
        add_rename(plan, artifact_resolved, output_target_for("artifact", artifact_resolved, clean_base), "output:artifact")

    raw_working = str(payload.get("working_media_path") or "").strip()
    raw_source = str(payload.get("source_media_path") or "").strip()
    raw_working_path = Path(raw_working).expanduser() if raw_working else None
    raw_source_path = Path(raw_source).expanduser() if raw_source else None
    if raw_working_path and raw_source_path and not raw_working_path.exists() and raw_source_path.exists():
        add_pointer_replacement(plan, raw_working_path, raw_source_path)
        media_paths = []
    else:
        media_paths = existing_media_paths(payload)
    for media_path in media_paths:
        media_target = media_path.with_name(f"{clean_base}{media_path.suffix}")
        if media_path.resolve() != media_target.resolve() and not media_name_needs_cleanup(media_path):
            continue
        if media_path.resolve() != media_target.resolve() and media_name_needs_cleanup(media_target):
            plan.reason = "canonical media target still contains cleanup noise"
            continue
        add_rename(plan, media_path, media_target, "media")

    if plan.skipped:
        return plan
    if not plan.operations and not plan.replacements:
        plan.skipped = True
        plan.reason = "already canonical"
    return plan


def replace_strings(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, str):
        return replacements.get(value, value)
    if isinstance(value, list):
        return [replace_strings(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: replace_strings(item, replacements) for key, item in value.items()}
    return value


def rewrite_artifact_payload(path: Path, replacements: dict[str, str]) -> None:
    payload = load_json_file(path)
    media_targets = [new for old, new in replacements.items() if Path(old).suffix.lower() in AUDIO_VIDEO_EXTENSIONS]
    if media_targets:
        payload["source_media_path"] = media_targets[0]
        payload["working_media_path"] = media_targets[0]
    payload = replace_strings(payload, replacements)
    write_json_file(path, payload)


def rewrite_state_file(state_path: Path, replacements: dict[str, str]) -> bool:
    if not state_path.exists():
        return False
    payload = load_json(state_path)
    if not isinstance(payload, dict):
        return False
    jobs = payload.get("jobs")
    if not isinstance(jobs, dict):
        return False

    for job_state in jobs.values():
        if not isinstance(job_state, dict):
            continue
        for section_name in ("processed", "candidates"):
            section = job_state.get(section_name)
            if not isinstance(section, dict):
                continue
            rewritten: dict[str, Any] = {}
            for key, value in section.items():
                new_key = replacements.get(str(key), str(key))
                rewritten[new_key] = replace_strings(value, replacements)
            job_state[section_name] = rewritten

    dump_json(state_path, payload)
    return True


def service_is_active(service_name: str) -> bool:
    completed = subprocess.run(
        ["systemctl", "--user", "is-active", "--quiet", service_name],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


def stop_service(service_name: str) -> bool:
    was_active = service_is_active(service_name)
    if was_active:
        subprocess.run(["systemctl", "--user", "stop", service_name], check=True)
    return was_active


def start_service(service_name: str) -> None:
    subprocess.run(["systemctl", "--user", "start", service_name], check=True)


def apply_plan(plan: CleanupPlan) -> dict[str, Any]:
    applied: list[dict[str, str]] = []
    for operation in plan.operations:
        operation.new_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(operation.old_path), str(operation.new_path))
        applied.append(operation.to_dict())

    artifact_path = Path(plan.replacements.get(str(plan.artifact_path), str(plan.artifact_path)))
    rewrite_artifact_payload(artifact_path, plan.replacements)
    return {"artifact_path": str(artifact_path), "applied": applied}


def quarantine_path_for(root: Path, source_path: Path) -> Path:
    digest = sha256_file(source_path)[:16] if source_path.exists() else "missing"
    return root / digest[:2] / f"{digest}-{source_path.name}"


def resolve_identical_conflict_plan(plan: CleanupPlan, *, quarantine_dir: Path) -> Optional[dict[str, Any]]:
    if not plan.skipped:
        return None

    conflicts = target_conflicts(plan)
    if not conflicts:
        return None
    equivalent_conflicts: list[dict[str, str]] = []
    for conflict in conflicts:
        old_path = Path(conflict["old_path"])
        target_path = Path(conflict["target_path"])
        role = conflict["role"]
        if not content_equivalent(old_path, target_path, role):
            return None
        equivalent_conflicts.append(conflict)

    replacements = dict(plan.replacements)
    quarantined: list[dict[str, str]] = []
    root = quarantine_dir.expanduser().resolve()
    for conflict in equivalent_conflicts:
        old_path = Path(conflict["old_path"])
        target_path = Path(conflict["target_path"])
        quarantine_path = quarantine_path_for(root, old_path)
        quarantine_path.parent.mkdir(parents=True, exist_ok=True)
        if quarantine_path.exists():
            raise CleanupError(f"Quarantine target already exists: {quarantine_path}")
        shutil.move(str(old_path), str(quarantine_path))
        replacements[str(old_path.resolve())] = str(target_path.resolve())
        quarantined.append(
            {
                "role": conflict["role"],
                "old_path": str(old_path),
                "canonical_path": str(target_path),
                "quarantine_path": str(quarantine_path),
            }
        )

    media_moves: list[dict[str, str]] = []
    for operation in plan.operations:
        if operation.role.startswith("output:"):
            continue
        if operation.new_path.exists():
            return None
        operation.new_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(operation.old_path), str(operation.new_path))
        replacements[str(operation.old_path)] = str(operation.new_path)
        media_moves.append(operation.to_dict())

    artifact_path = replacements.get(str(plan.artifact_path.resolve()), str(plan.artifact_path.resolve()))
    if Path(artifact_path).exists():
        rewrite_artifact_payload(Path(artifact_path), replacements)
    return {
        "artifact_path": artifact_path,
        "resolved_identical_conflict": True,
        "quarantined": quarantined,
        "media_moves": media_moves,
        "replacements": replacements,
    }


def refresh_store_for_replacements(
    replacements: dict[str, str],
    artifact_paths: Iterable[str],
    *,
    store_dir: Path,
    embedding_provider: str,
    embedding_model: str,
) -> list[dict[str, str]]:
    from transcript_store import connect, ingest_artifact, init_db

    refreshed: list[dict[str, str]] = []
    artifact_replacements = {
        old: new
        for old, new in replacements.items()
        if old.endswith(".transcript.json") and new.endswith(".transcript.json") and old != new
    }
    new_artifact_paths = sorted({str(Path(path).expanduser().resolve()) for path in artifact_paths})
    if not artifact_replacements and not new_artifact_paths:
        return refreshed

    root = store_dir.expanduser()
    with connect(root) as con:
        init_db(con)
        stale_paths = sorted({*artifact_replacements.keys(), *new_artifact_paths})
        for old_path in stale_paths:
            row = con.execute(
                "SELECT id FROM documents WHERE kind = 'transcript' AND source_path = ?",
                (old_path,),
            ).fetchone()
            if row:
                doc_id = str(row["id"])
                con.execute("DELETE FROM documents_fts WHERE doc_id = ?", (doc_id,))
                con.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        con.commit()

    for new_path in new_artifact_paths:
        result = ingest_artifact(
            Path(new_path),
            root=root,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
        )
        refreshed.append(
            {
                "new_source_path": result.source_path,
                "document_id": result.id,
                "status": result.status,
            }
        )
    return refreshed


def summarize(plans: list[CleanupPlan]) -> dict[str, int]:
    actionable = [plan for plan in plans if not plan.skipped and (plan.operations or plan.replacements)]
    return {
        "scanned": len(plans),
        "actionable": len(actionable),
        "skipped": len([plan for plan in plans if plan.skipped]),
        "operations": sum(len(plan.operations) for plan in actionable),
    }


def build_review_entry(plan: CleanupPlan) -> dict[str, Any]:
    payload = load_json_file(plan.artifact_path)
    output_paths = payload.get("output_paths") if isinstance(payload.get("output_paths"), dict) else {}
    conflicts = []
    for conflict in target_conflicts(plan):
        old_path = Path(conflict["old_path"])
        target_path = Path(conflict["target_path"])
        role = conflict["role"]
        conflicts.append(
            {
                **conflict,
                "old_sha256": sha256_file(old_path) if old_path.exists() else "",
                "target_sha256": sha256_file(target_path) if target_path.exists() else "",
                "content_equivalent": content_equivalent(old_path, target_path, role),
            }
        )
    return {
        "artifact_path": str(plan.artifact_path),
        "reason": plan.reason,
        "suggested_action": suggested_review_action(plan),
        "clean_base_name": plan.clean_base_name,
        "event": event_review(payload),
        "source_media_path": payload.get("source_media_path"),
        "working_media_path": payload.get("working_media_path"),
        "output_paths": output_paths,
        "operations": [operation.to_dict() for operation in plan.operations],
        "replacements": plan.replacements,
        "target_conflicts": conflicts,
    }


def build_review_payload(plans: list[CleanupPlan]) -> dict[str, Any]:
    skipped = [plan for plan in plans if plan.skipped]
    by_reason: dict[str, int] = {}
    for plan in skipped:
        reason = plan.reason or "unspecified"
        by_reason[reason] = by_reason.get(reason, 0) + 1
    return {
        "schema_version": 1,
        "summary": {
            **summarize(plans),
            "review_count": len(skipped),
            "by_reason": by_reason,
        },
        "items": [build_review_entry(plan) for plan in skipped],
    }


def write_review_file(path: Path, plans: list[CleanupPlan]) -> Path:
    output_path = path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_file(output_path, build_review_payload(plans))
    return output_path


def suppress_conflicting_media_operations(plans: list[CleanupPlan]) -> None:
    media_targets: dict[str, set[str]] = {}
    for plan in plans:
        if plan.skipped:
            continue
        for operation in plan.operations:
            if operation.role != "media":
                continue
            media_targets.setdefault(str(operation.old_path), set()).add(str(operation.new_path))

    conflicting_old_paths = {old for old, targets in media_targets.items() if len(targets) > 1}
    if not conflicting_old_paths:
        return

    for plan in plans:
        if plan.skipped:
            continue
        kept_operations: list[RenameOperation] = []
        removed = False
        for operation in plan.operations:
            if operation.role == "media" and str(operation.old_path) in conflicting_old_paths:
                plan.replacements.pop(str(operation.old_path), None)
                removed = True
                continue
            kept_operations.append(operation)
        plan.operations = kept_operations
        if removed and not plan.reason:
            plan.reason = "shared media has conflicting canonical targets"
        if removed and not plan.operations and not plan.replacements:
            plan.skipped = True


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    artifacts = iter_artifact_paths(args.roots, recursive=args.recursive)
    plans = [plan_artifact_cleanup(path) for path in artifacts]
    suppress_conflicting_media_operations(plans)
    plans = [plan for plan in plans if not plan.skipped or plan.operations]
    if args.limit is not None:
        limited: list[CleanupPlan] = []
        included_actionable = 0
        for plan in plans:
            if plan.skipped or not (plan.operations or plan.replacements):
                limited.append(plan)
            elif included_actionable < args.limit:
                limited.append(plan)
                included_actionable += 1
        plans = limited

    result: dict[str, Any] = {"summary": summarize(plans), "plans": [plan.to_dict() for plan in plans]}
    if args.export_review:
        review_path = write_review_file(args.export_review, plans)
        result["review_path"] = str(review_path)
    if not args.apply:
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            summary = result["summary"]
            print(
                f"Dry run: scanned={summary['scanned']} actionable={summary['actionable']} "
                f"operations={summary['operations']} skipped={summary['skipped']}"
            )
            for plan in plans:
                if plan.skipped:
                    continue
                print(f"\n{plan.artifact_path}")
                for operation in plan.operations:
                    print(f"  {operation.role}: {operation.old_path.name} -> {operation.new_path.name}")
            if args.export_review:
                print(f"Review file: {result['review_path']}")
        return 0

    if service_is_active(args.service_name) and not args.manage_service:
        print(
            f"Error: {args.service_name} is active. Re-run with --manage-service to stop/restart it during apply.",
            file=sys.stderr,
        )
        return 2

    service_was_active = False
    try:
        if args.manage_service:
            service_was_active = stop_service(args.service_name)
        replacements: dict[str, str] = {}
        applied_results: list[dict[str, Any]] = []
        for plan in plans:
            if args.resolve_identical_conflicts and plan.skipped:
                resolved_result = resolve_identical_conflict_plan(plan, quarantine_dir=args.quarantine_dir)
                if resolved_result:
                    applied_results.append(resolved_result)
                    replacements.update(resolved_result["replacements"])
                continue
            if plan.skipped or not (plan.operations or plan.replacements):
                continue
            applied_result = apply_plan(plan)
            applied_results.append(applied_result)
            replacements.update(plan.replacements)
        state_updated = rewrite_state_file(args.state_file.expanduser().resolve(), replacements) if replacements else False
        store_refreshed = (
            refresh_store_for_replacements(
                replacements,
                [item["artifact_path"] for item in applied_results],
                store_dir=args.store_dir,
                embedding_provider=args.embedding_provider,
                embedding_model=args.embedding_model,
            )
            if args.refresh_store
            else []
        )
        result["applied"] = applied_results
        result["state_updated"] = state_updated
        result["store_refreshed"] = store_refreshed
    finally:
        if args.manage_service and service_was_active:
            start_service(args.service_name)

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        summary = result["summary"]
        print(
            f"Applied cleanup: actionable={summary['actionable']} operations={summary['operations']} "
            f"state_updated={result.get('state_updated', False)} "
            f"store_refreshed={len(result.get('store_refreshed', []))}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
