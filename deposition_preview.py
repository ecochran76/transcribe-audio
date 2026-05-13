#!/usr/bin/env python3
"""
Create preview-only deposition and memory-harvest plans from readout artifacts.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

from deposition_artifacts import DepositAction, DepositPreview, MemoryHarvestCandidate, normalize_string
from transcribe_common import TranscriptionError

DEPOSITION_PREVIEW_JSON_STDOUT_PREFIX = "DEPOSITION_PREVIEW_JSON="


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a dry-run deposition and memory-harvest preview from a readout artifact."
    )
    parser.add_argument("readout", type=Path, help="Path to a *.readout.json or *.contextual.readout.json artifact.")
    parser.add_argument("--route", type=Path, help="Path to the source *.route.json decision artifact.")
    parser.add_argument("--transcript", type=Path, help="Path to the source *.transcript.json artifact.")
    parser.add_argument("--output-dir", type=Path, help="Directory for preview output. Defaults beside readout.")
    parser.add_argument(
        "--local-root",
        type=Path,
        help="Preview a local filesystem deposition under this root. No files are copied.",
    )
    parser.add_argument("--drive-folder-id", help="Preview a Google Drive deposition target folder id.")
    parser.add_argument("--drive-profile", help="Drive/gws/gog profile label to record in the preview.")
    parser.add_argument("--odoo-profile", help="Preview an Odoo/Odollo deposition profile.")
    parser.add_argument("--odoo-model", help="Preview an Odoo target model, for example crm.lead.")
    parser.add_argument("--odoo-record-id", help="Preview an Odoo target record id.")
    parser.add_argument("--graphiti-group", default="transcribe_audio_main", help="Graphiti memory target group id.")
    parser.add_argument(
        "--include-transcript",
        action="store_true",
        help="Include transcript artifact path in local/Drive deposition actions. Memory harvest still excludes raw transcript text.",
    )
    return parser.parse_args(argv)


def load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    except OSError as exc:
        raise TranscriptionError(f"Failed to read {label} {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise TranscriptionError(f"{label} {path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise TranscriptionError(f"{label} {path} must contain a JSON object.")
    return payload


def output_base_path(readout_path: Path, output_dir: Optional[Path]) -> Path:
    base_name = readout_path.name
    for suffix in (".contextual.readout.json", ".readout.json"):
        if base_name.endswith(suffix):
            base_name = base_name[: -len(suffix)]
            break
    else:
        base_name = readout_path.stem
    directory = output_dir.expanduser() if output_dir else readout_path.parent
    return directory / f"{base_name}.deposit-preview.json"


def compact_selected_candidate(route: dict[str, Any]) -> dict[str, Any]:
    selected = route.get("selected_candidate") if isinstance(route.get("selected_candidate"), dict) else {}
    return {
        "label": selected.get("label"),
        "target_kind": selected.get("target_kind"),
        "target_id": selected.get("target_id"),
        "confidence": selected.get("confidence"),
        "source": selected.get("source"),
        "provenance_source_ids": selected.get("provenance_source_ids") or [],
    }


def supporting_source_ids(readout: dict[str, Any]) -> list[str]:
    contextualization = readout.get("contextualization") if isinstance(readout.get("contextualization"), dict) else {}
    sources = contextualization.get("supporting_context_sources") if isinstance(contextualization, dict) else []
    source_ids = []
    for source in sources or []:
        if not isinstance(source, dict):
            continue
        source_id = normalize_string(source.get("source_id") or source.get("label"))
        if source_id and source_id not in source_ids:
            source_ids.append(source_id)
    return source_ids


def preview_warnings(readout: dict[str, Any], route: dict[str, Any]) -> list[str]:
    values = []
    contextualization = readout.get("contextualization") if isinstance(readout.get("contextualization"), dict) else {}
    for item in contextualization.get("warnings") or []:
        text = normalize_string(item)
        if text:
            values.append(text)
    for item in route.get("warnings") or []:
        text = normalize_string(item)
        if text:
            values.append(text)
    excluded_count = contextualization.get("excluded_source_count")
    if isinstance(excluded_count, int) and excluded_count > 0:
        values.append(f"{excluded_count} provenance source(s) were excluded before deposition preview.")
    result = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def source_paths(
    *,
    readout_path: Path,
    route_path: Optional[Path],
    transcript_path: Optional[Path],
    include_transcript: bool,
) -> list[str]:
    paths = [str(readout_path)]
    if route_path:
        paths.append(str(route_path))
    if transcript_path and include_transcript:
        paths.append(str(transcript_path))
    return paths


def build_actions(
    *,
    args: argparse.Namespace,
    readout: dict[str, Any],
    readout_path: Path,
    route: dict[str, Any],
    route_path: Optional[Path],
    transcript_path: Optional[Path],
) -> list[DepositAction]:
    selected = compact_selected_candidate(route)
    evidence = []
    if normalize_string(selected.get("label")):
        evidence.append(f"selected route candidate: {selected['label']}")
    if readout.get("title"):
        evidence.append(f"readout: {readout['title']}")
    paths = source_paths(
        readout_path=readout_path,
        route_path=route_path,
        transcript_path=transcript_path,
        include_transcript=args.include_transcript,
    )
    actions: list[DepositAction] = []
    if args.local_root:
        actions.append(
            DepositAction(
                action_type="copy_artifacts",
                target_kind="local_filesystem",
                target_id=str(args.local_root.expanduser()),
                source_paths=paths,
                evidence=evidence,
                metadata={"selected_candidate": selected, "writes_enabled": False},
            )
        )
    if args.drive_folder_id:
        actions.append(
            DepositAction(
                action_type="upload_artifacts",
                target_kind="google_drive",
                target_id=args.drive_folder_id,
                target_profile=args.drive_profile or "",
                source_paths=paths,
                evidence=evidence,
                metadata={"selected_candidate": selected, "writes_enabled": False},
            )
        )
    if args.odoo_profile or args.odoo_model or args.odoo_record_id:
        actions.append(
            DepositAction(
                action_type="attach_readout_note",
                target_kind="odoo_record",
                target_id="/".join(
                    item
                    for item in [
                        normalize_string(args.odoo_model),
                        normalize_string(args.odoo_record_id),
                    ]
                    if item
                ),
                target_profile=args.odoo_profile or "",
                source_paths=[str(readout_path), *([str(route_path)] if route_path else [])],
                evidence=evidence,
                metadata={
                    "selected_candidate": selected,
                    "requires_target_model": not bool(args.odoo_model),
                    "requires_target_record_id": not bool(args.odoo_record_id),
                    "writes_enabled": False,
                },
            )
        )
    return actions


def build_memory_candidates(
    readout: dict[str, Any],
    *,
    readout_path: Path,
    graphiti_group: str,
) -> list[MemoryHarvestCandidate]:
    source_ids = supporting_source_ids(readout)
    candidates: list[MemoryHarvestCandidate] = []
    for item in readout.get("memory_candidates") or []:
        if not isinstance(item, dict):
            continue
        text = normalize_string(item.get("text") or item.get("memory"))
        if not text:
            continue
        candidates.append(
            MemoryHarvestCandidate(
                text=text,
                kind=normalize_string(item.get("kind")) or "memory",
                evidence=normalize_string(item.get("evidence")),
                target_group_id=graphiti_group,
                source_readout_path=str(readout_path),
                source_ids=source_ids,
            )
        )
    return candidates


def generate_deposition_preview(args: argparse.Namespace) -> Path:
    readout_path = args.readout.expanduser().resolve()
    route_path = args.route.expanduser().resolve() if args.route else None
    transcript_path = args.transcript.expanduser().resolve() if args.transcript else None
    readout = load_json_object(readout_path, "readout artifact")
    route = load_json_object(route_path, "route decision") if route_path else {}
    preview = DepositPreview(
        source_readout_path=str(readout_path),
        source_route_path=str(route_path) if route_path else "",
        source_transcript_path=str(transcript_path) if transcript_path else "",
        selected_candidate=compact_selected_candidate(route),
        review_required=bool(route.get("review_required", True)),
        warnings=preview_warnings(readout, route),
        actions=build_actions(
            args=args,
            readout=readout,
            readout_path=readout_path,
            route=route,
            route_path=route_path,
            transcript_path=transcript_path,
        ),
        memory_candidates=build_memory_candidates(
            readout,
            readout_path=readout_path,
            graphiti_group=args.graphiti_group,
        ),
    )
    output_path = output_base_path(readout_path, args.output_dir)
    preview.write_json(output_path)
    return output_path


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    try:
        output_path = generate_deposition_preview(args)
    except TranscriptionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Writing deposition preview JSON to {output_path}...")
    print(f"{DEPOSITION_PREVIEW_JSON_STDOUT_PREFIX}{output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
