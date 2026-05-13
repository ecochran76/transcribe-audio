#!/usr/bin/env python3
"""
Backfill calendar metadata and calendar-based filenames for transcript artifacts.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Optional

from transcript_artifacts import json_ready
from transcribe_common import (
    CalendarProviderConfig,
    TranscriptionError,
    build_calendar_service,
    build_event_base_name,
    extract_event_metadata,
    find_matching_events,
    format_utterance,
    write_docx,
    write_text,
)
from watch_transcriptions import DEFAULT_STATE_PATH, dump_json, fingerprint_for, load_json


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair recent transcript sidecars/outputs that missed calendar metadata."
    )
    parser.add_argument(
        "--artifact-glob",
        default=str(Path.home() / "Downloads" / "*.transcript.json"),
        help="Glob for transcript artifacts to consider.",
    )
    parser.add_argument("--since-days", type=float, default=4.0, help="Only consider artifacts modified this many days ago.")
    parser.add_argument("--calendar-id", default="primary", help="Calendar ID to query.")
    parser.add_argument("--calendar-window", type=float, default=24.0, help="Calendar lookup window in hours.")
    parser.add_argument(
        "--calendar-providers",
        default="gog,gws",
        help="Comma-separated provider list for repair. Default avoids Google OAuth fallback.",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=DEFAULT_STATE_PATH,
        help="Watcher state file to update when media/artifact paths are renamed.",
    )
    parser.add_argument("--apply", action="store_true", help="Apply changes. Without this, only prints a plan.")
    parser.add_argument(
        "--rename-media",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rename source media files and update watcher state (default: true).",
    )
    return parser.parse_args(argv)


def parse_datetime(value: str) -> datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value)


def load_artifact(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TranscriptionError(f"{path} is not a JSON object.")
    return payload


def normalize_existing_event_info(event_info: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(event_info)
    for field in ("start", "end"):
        value = normalized.get(field)
        if isinstance(value, str) and value:
            try:
                normalized[field] = parse_datetime(value)
            except ValueError:
                pass
    return normalized


def provider_configs(raw_value: str) -> list[CalendarProviderConfig]:
    configs: list[CalendarProviderConfig] = []
    for item in raw_value.split(","):
        name = item.strip()
        if name:
            configs.append(CalendarProviderConfig(name=name))
    if not configs:
        raise TranscriptionError("At least one calendar provider is required.")
    return configs


def candidate_artifacts(pattern: str, since_days: float) -> list[Path]:
    cutoff = time.time() - since_days * 24 * 60 * 60
    paths = [Path(path) for path in glob.glob(pattern)]
    return sorted(path for path in paths if path.stat().st_mtime >= cutoff)


def unique_path(path: Path, *, existing_ok: bool = False) -> Path:
    if existing_ok or not path.exists():
        return path
    raise TranscriptionError(f"Refusing to overwrite existing path: {path}")


def build_repair_plan(
    artifact_path: Path,
    artifact: dict[str, Any],
    *,
    calendar_service: Any,
    calendar_id: str,
    calendar_window: float,
    rename_media: bool,
) -> Optional[dict[str, Any]]:
    existing_event_info = artifact.get("event") if isinstance(artifact.get("event"), dict) else None
    if existing_event_info and existing_event_info.get("matching_calendars"):
        return None

    recording_start = parse_datetime(str(artifact["recording_start"]))
    recording_end = parse_datetime(str(artifact["recording_end"]))
    matching_events, fallback_event, matching_calendars = find_matching_events(
        calendar_service,
        calendar_id,
        recording_start,
        recording_end,
        calendar_window,
    )
    event = matching_events[0]["event"] if matching_events else fallback_event
    if not event and not existing_event_info:
        return {
            "artifact_path": artifact_path,
            "status": "no_event",
        }

    event_info = normalize_existing_event_info(existing_event_info) if existing_event_info else extract_event_metadata(event)
    event_info["matching_calendars"] = matching_calendars
    media_field = "working_media_path" if existing_event_info else "source_media_path"
    source_media = Path(str(artifact.get(media_field) or artifact.get("working_media_path") or ""))
    old_outputs = artifact.get("output_paths") or {}

    if existing_event_info:
        new_media = source_media
        new_outputs = {
            "docx": old_outputs.get("docx", str(artifact_path.with_suffix(".docx"))),
            "txt": old_outputs.get("txt", str(artifact_path.with_suffix(".txt"))),
            "artifact": old_outputs.get("artifact", str(artifact_path)),
        }
    else:
        event_start = event_info.get("start") or recording_start
        source_stem = source_media.stem if source_media.name else artifact_path.name.removesuffix(" Transcript.transcript.json")
        new_base = build_event_base_name(event_start, event_info.get("summary", "Untitled Event"), source_stem)
        directory = artifact_path.parent
        output_base = directory / f"{new_base} Transcript"
        new_media = source_media.with_name(f"{new_base}{source_media.suffix}") if rename_media and source_media.exists() else source_media
        new_outputs = {
            "docx": str(output_base.with_suffix(".docx")),
            "txt": str(output_base.with_suffix(".txt")),
            "artifact": str(output_base.with_suffix(".transcript.json")),
        }

    return {
        "artifact_path": artifact_path,
        "artifact": artifact,
        "status": "matched",
        "event_info": event_info,
        "old_media": source_media,
        "new_media": new_media,
        "old_outputs": old_outputs,
        "new_outputs": new_outputs,
    }


def update_state(state_path: Path, plans: list[dict[str, Any]]) -> None:
    if not state_path.exists():
        return

    payload = load_json(state_path)
    jobs_payload = payload.get("jobs") if isinstance(payload, dict) else None
    if not isinstance(jobs_payload, dict):
        return

    for plan in plans:
        old_media = str(plan["old_media"])
        new_media = str(plan["new_media"])
        new_outputs = plan["new_outputs"]
        for job_payload in jobs_payload.values():
            if not isinstance(job_payload, dict):
                continue
            processed = job_payload.get("processed")
            if not isinstance(processed, dict) or old_media not in processed:
                continue
            record = processed.pop(old_media)
            record["artifact_paths"] = [new_outputs["artifact"]]
            record["command"] = [new_media if item == old_media else item for item in record.get("command") or []]
            try:
                media_path = Path(new_media)
                stats = media_path.stat()
                record["size"] = int(stats.st_size)
                record["mtime"] = float(stats.st_mtime)
                record["fingerprint"] = fingerprint_for(media_path, int(stats.st_size), float(stats.st_mtime))
            except OSError:
                pass
            processed[new_media] = record

    dump_json(state_path, payload)


def apply_plan(plan: dict[str, Any]) -> None:
    artifact = dict(plan["artifact"])
    event_info = plan["event_info"]
    old_media = Path(plan["old_media"])
    new_media = Path(plan["new_media"])
    old_outputs = plan["old_outputs"]
    new_outputs = plan["new_outputs"]

    if old_media.exists() and new_media != old_media:
        unique_path(new_media)
        old_media.rename(new_media)

    utterances = artifact.get("utterances") or []
    docx_path = unique_path(Path(new_outputs["docx"]), existing_ok=Path(old_outputs.get("docx", "")) == Path(new_outputs["docx"]))
    txt_path = unique_path(Path(new_outputs["txt"]), existing_ok=Path(old_outputs.get("txt", "")) == Path(new_outputs["txt"]))
    artifact_new_path = unique_path(
        Path(new_outputs["artifact"]),
        existing_ok=Path(old_outputs.get("artifact", "")) == Path(new_outputs["artifact"]),
    )

    write_docx(utterances, docx_path, event_info=event_info, title=artifact.get("transcript_title") or "Transcript")
    write_text(utterances, txt_path, event_info=event_info)

    artifact["event"] = event_info
    artifact["working_media_path"] = str(new_media)
    artifact["output_paths"] = new_outputs
    with artifact_new_path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(artifact), handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")

    for old_path_raw in old_outputs.values():
        old_path = Path(str(old_path_raw))
        if old_path.exists() and old_path not in {docx_path, txt_path, artifact_new_path}:
            old_path.unlink()


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    try:
        service = build_calendar_service(
            Path("credentials.json"),
            Path("token.json"),
            provider_configs=provider_configs(args.calendar_providers),
        )
        plans = []
        for artifact_path in candidate_artifacts(args.artifact_glob, args.since_days):
            artifact = load_artifact(artifact_path)
            plan = build_repair_plan(
                artifact_path,
                artifact,
                calendar_service=service,
                calendar_id=args.calendar_id,
                calendar_window=args.calendar_window,
                rename_media=args.rename_media,
            )
            if plan:
                plans.append(plan)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    matched = [plan for plan in plans if plan.get("status") == "matched"]
    no_event = [plan for plan in plans if plan.get("status") == "no_event"]
    for plan in matched:
        event = plan["event_info"]
        print(f"MATCH {plan['artifact_path']}")
        print(f"  event: {event.get('summary')} ({event.get('start')} - {event.get('end')})")
        print(f"  media: {plan['old_media']} -> {plan['new_media']}")
        print(f"  artifact: {plan['new_outputs']['artifact']}")
    for plan in no_event:
        print(f"NO_EVENT {plan['artifact_path']}")

    if not args.apply:
        print(f"Dry run: {len(matched)} matched, {len(no_event)} without event. Re-run with --apply to modify files.")
        return 0

    try:
        for plan in matched:
            apply_plan(plan)
        update_state(args.state_file.expanduser().resolve(), matched)
    except Exception as exc:
        print(f"Error while applying repairs: {exc}", file=sys.stderr)
        return 1

    print(f"Applied repairs: {len(matched)} matched, {len(no_event)} without event.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
