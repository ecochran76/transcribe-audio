#!/usr/bin/env python3
"""
Dry-run matter routing for transcript/readout artifacts.
"""
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

from context_sources import (
    GraphitiProvenanceConfig,
    GwsProvenanceConfig,
    OdolloProvenanceConfig,
    collect_graphiti_provenance,
    collect_gws_provenance,
    collect_odollo_provenance,
    filter_provenance_sources,
)
from routing_artifacts import (
    ReviewQueueItem,
    build_route_decision,
    candidates_from_graphiti_sources,
    stable_id,
)
from transcribe_common import TranscriptionError

ROUTE_JSON_STDOUT_PREFIX = "ROUTE_DECISION_JSON="


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an auditable dry-run route decision JSON.")
    parser.add_argument("transcript", type=Path, help="Path to a *.transcript.json artifact.")
    parser.add_argument("readout", type=Path, help="Path to a *.readout.json artifact.")
    parser.add_argument("--output-dir", type=Path, help="Directory for route decision output. Defaults beside readout.")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.8,
        help="Minimum selected-candidate confidence before bypassing the review queue.",
    )
    parser.add_argument(
        "--review-queue-dir",
        type=Path,
        default=Path("~/.local/state/transcribe-audio/review-queue"),
        help="Local review queue directory for low-confidence route decisions.",
    )
    parser.add_argument(
        "--no-review-queue",
        action="store_true",
        help="Do not write a local review queue item even when review is required.",
    )
    parser.add_argument("--gws-provenance", action="store_true", help="Collect read-only provenance through gws.")
    parser.add_argument(
        "--gws-config-dir",
        type=Path,
        help="Set GOOGLE_WORKSPACE_CLI_CONFIG_DIR for gws provenance calls.",
    )
    parser.add_argument("--gws-drive-query", help="Explicit Google Drive query for gws provenance.")
    parser.add_argument("--gws-drive-page-size", type=int, default=5, help="Maximum Drive files to collect.")
    parser.add_argument("--gws-timeout", type=float, default=30.0, help="Timeout per gws command.")
    parser.add_argument(
        "--no-gws-calendar-details",
        action="store_true",
        help="Skip live gws calendar event detail lookups.",
    )
    parser.add_argument("--no-gws-drive", action="store_true", help="Skip live gws Drive search.")
    parser.add_argument(
        "--graphiti-provenance",
        action="store_true",
        help="Collect read-only advisory provenance and candidate hints through graphiti-runtime.",
    )
    parser.add_argument(
        "--graphiti-group",
        action="append",
        dest="graphiti_groups",
        help="Graphiti group_id to query. Repeatable. Defaults to transcribe_audio_main.",
    )
    parser.add_argument(
        "--graphiti-command",
        default=str(Path("~/.local/bin/graphiti-runtime").expanduser()),
        help="Path to graphiti-runtime.",
    )
    parser.add_argument("--graphiti-timeout", type=float, default=30.0, help="Timeout per Graphiti discovery call.")
    parser.add_argument("--graphiti-max-facts", type=int, default=5, help="Maximum Graphiti facts per group.")
    parser.add_argument("--graphiti-max-nodes", type=int, default=5, help="Maximum Graphiti nodes per group.")
    parser.add_argument("--graphiti-max-episodes", type=int, default=5, help="Maximum Graphiti episodes per group.")
    parser.add_argument(
        "--no-graphiti-candidates",
        action="store_true",
        help="Add Graphiti provenance sources but do not add advisory route candidates.",
    )
    parser.add_argument(
        "--odollo-provenance",
        action="store_true",
        help="Collect read-only Odollo/Odoo contact and log-note provenance.",
    )
    parser.add_argument(
        "--odollo-profile",
        action="append",
        dest="odollo_profiles",
        help="Odollo profile to query. Repeatable. Defaults to soylei-prod and saber-prod.",
    )
    parser.add_argument(
        "--odollo-command",
        default=str(Path("~/workspace.local/odollo/.venv/bin/python").expanduser()) + " -m odollo.cli",
        help="Command prefix for Odollo CLI.",
    )
    parser.add_argument(
        "--odollo-repo",
        type=Path,
        default=Path("~/workspace.local/odollo"),
        help="Odollo repository root for CLI execution.",
    )
    parser.add_argument(
        "--odollo-config",
        type=Path,
        default=Path("~/.odollo/odollo.yml"),
        help="Odollo config YAML path.",
    )
    parser.add_argument("--odollo-timeout", type=float, default=30.0, help="Timeout per Odollo command.")
    parser.add_argument("--odollo-limit", type=int, default=5, help="Maximum Odollo rows per model/profile.")
    parser.add_argument("--no-odollo-contacts", action="store_true", help="Skip Odollo contact searches.")
    parser.add_argument("--no-odollo-log-notes", action="store_true", help="Skip Odollo log-note searches.")
    parser.add_argument(
        "--provenance-quality-threshold",
        type=int,
        default=2,
        help="Minimum compact-term matches required for non-calendar provenance sources.",
    )
    parser.add_argument(
        "--no-provenance-quality-filter",
        action="store_true",
        help="Keep all collected provenance sources, recording them as unfiltered.",
    )
    return parser.parse_args(argv)


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    except OSError as exc:
        raise TranscriptionError(f"Failed to read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise TranscriptionError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise TranscriptionError(f"{path} must contain a JSON object.")
    return payload


def output_base_path(readout_path: Path, output_dir: Optional[Path]) -> Path:
    base_name = readout_path.name
    if base_name.endswith(".readout.json"):
        base_name = base_name[: -len(".readout.json")]
    else:
        base_name = readout_path.stem
    directory = output_dir.expanduser() if output_dir else readout_path.parent
    return directory / base_name


def review_queue_path(review_queue_dir: Path, route_decision_path: Path, selected_label: str) -> Path:
    queue_id = stable_id(str(route_decision_path), selected_label or "unselected")
    return review_queue_dir.expanduser() / f"{queue_id}.route-review.json"


def generate_route(args: argparse.Namespace) -> tuple[Path, Optional[Path]]:
    transcript_path = args.transcript.expanduser().resolve()
    readout_path = args.readout.expanduser().resolve()
    transcript = load_json_object(transcript_path)
    readout = load_json_object(readout_path)
    gws_sources = collect_gws_provenance(
        transcript,
        readout,
        config=GwsProvenanceConfig(
            enabled=args.gws_provenance,
            config_dir=args.gws_config_dir,
            drive_query=args.gws_drive_query or "",
            drive_page_size=args.gws_drive_page_size,
            timeout=args.gws_timeout,
            include_calendar_details=not args.no_gws_calendar_details,
            include_drive_search=not args.no_gws_drive,
        ),
    )
    graphiti_sources = collect_graphiti_provenance(
        transcript,
        readout,
        config=GraphitiProvenanceConfig(
            enabled=args.graphiti_provenance,
            group_ids=tuple(args.graphiti_groups or ["transcribe_audio_main"]),
            command=args.graphiti_command,
            timeout=args.graphiti_timeout,
            max_facts=args.graphiti_max_facts,
            max_nodes=args.graphiti_max_nodes,
            max_episodes=args.graphiti_max_episodes,
        ),
    )
    odollo_sources = collect_odollo_provenance(
        transcript,
        readout,
        config=OdolloProvenanceConfig(
            enabled=args.odollo_provenance,
            profiles=tuple(args.odollo_profiles or ["soylei-prod", "saber-prod"]),
            command=tuple(shlex.split(args.odollo_command)),
            repo_root=args.odollo_repo,
            config_path=args.odollo_config,
            timeout=args.odollo_timeout,
            limit=args.odollo_limit,
            include_contacts=not args.no_odollo_contacts,
            include_log_notes=not args.no_odollo_log_notes,
        ),
    )
    retained_sources, excluded_sources, provenance_warnings = filter_provenance_sources(
        gws_sources + graphiti_sources + odollo_sources,
        transcript=transcript,
        readout=readout,
        min_score=args.provenance_quality_threshold,
        enabled=not args.no_provenance_quality_filter,
    )
    retained_graphiti_sources = [source for source in retained_sources if source.source_type.startswith("graphiti_")]
    graphiti_candidates = (
        [] if args.no_graphiti_candidates else candidates_from_graphiti_sources(retained_graphiti_sources)
    )

    decision = build_route_decision(
        transcript_path=transcript_path,
        readout_path=readout_path,
        transcript=transcript,
        readout=readout,
        confidence_threshold=args.confidence_threshold,
        extra_provenance_sources=retained_sources,
        excluded_provenance_sources=excluded_sources,
        provenance_warnings=provenance_warnings,
        extra_candidates=graphiti_candidates,
    )
    route_path = output_base_path(readout_path, args.output_dir).with_suffix(".route.json")
    decision.write_json(route_path)

    review_path = None
    if decision.review_required and not args.no_review_queue:
        selected_label = decision.selected_candidate.label if decision.selected_candidate else ""
        review_item = ReviewQueueItem(
            route_decision_path=str(route_path),
            selected_label=selected_label,
            reason="Route confidence below threshold; human review required before deposition.",
        )
        review_path = review_queue_path(args.review_queue_dir, route_path, selected_label)
        review_item.write_json(review_path)

    return route_path, review_path


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    try:
        route_path, review_path = generate_route(args)
    except TranscriptionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Writing route decision JSON to {route_path}...")
    print(f"{ROUTE_JSON_STDOUT_PREFIX}{route_path}")
    if review_path:
        print(f"Writing review queue item to {review_path}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
