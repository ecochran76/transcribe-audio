#!/usr/bin/env python3
"""
Generate an upgraded readout from a transcript, prior readout, and route decision.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

import summarize_transcript
from readout_artifacts import Readout
from transcript_store import TranscriptStoreError, ingest_artifact
from transcribe_common import TranscriptionError

CONTEXTUAL_READOUT_JSON_STDOUT_PREFIX = "CONTEXTUAL_READOUT_JSON="


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a contextual reread from transcript, readout, and route decision artifacts."
    )
    parser.add_argument("transcript", type=Path, help="Path to a *.transcript.json artifact.")
    parser.add_argument("readout", type=Path, help="Path to the prior *.readout.json artifact.")
    parser.add_argument("route", type=Path, help="Path to the *.route.json decision artifact.")
    parser.add_argument("--output-dir", type=Path, help="Directory for contextual readout outputs. Defaults beside route.")
    parser.add_argument(
        "--provider",
        choices=("openai-compatible", "codex-exec"),
        default="openai-compatible",
        help="Readout intelligence provider.",
    )
    parser.add_argument("--model", default=summarize_transcript.DEFAULT_OPENAI_MODEL, help="Model name.")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible API base URL.")
    parser.add_argument(
        "--api-key-file",
        default="api_keys.json",
        help="Path to JSON file containing openai_api_key and optionally openai_base_url.",
    )
    parser.add_argument("--openai-api-key", dest="openai_api_key", help="OpenAI-compatible API key.")
    parser.add_argument("--openai-api-key-stdin", action="store_true", help="Read API key from stdin.")
    parser.add_argument("--openai-api-key-prompt", action="store_true", help="Prompt for API key interactively.")
    parser.add_argument("--timeout", type=float, default=120.0, help="Provider request timeout in seconds.")
    parser.add_argument("--temperature", type=float, default=0.1, help="Provider sampling temperature.")
    parser.add_argument("--max-sources", type=int, default=12, help="Maximum supporting context sources to include.")
    parser.add_argument("--snippet-chars", type=int, default=500, help="Maximum snippet characters per source.")
    parser.add_argument("--store", action="store_true", help="Ingest the generated contextual readout into ~/.transcripts.")
    parser.add_argument("--store-dir", type=Path, help="Override transcript store directory.")
    parser.add_argument(
        "--all-provenance",
        action="store_true",
        help="Include all route provenance instead of the selected candidate's cited sources plus calendar context.",
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


def output_base_path(route_path: Path, output_dir: Optional[Path]) -> Path:
    base_name = route_path.name
    if base_name.endswith(".route.json"):
        base_name = base_name[: -len(".route.json")]
    else:
        base_name = route_path.stem
    directory = output_dir.expanduser() if output_dir else route_path.parent
    return directory / f"{base_name}.contextual"


def normalize_string(value: Any) -> str:
    return str(value or "").strip()


def source_key(source: dict[str, Any]) -> str:
    return normalize_string(source.get("source_id")) or normalize_string(source.get("label"))


def selected_source_ids(route: dict[str, Any]) -> set[str]:
    selected = route.get("selected_candidate") if isinstance(route.get("selected_candidate"), dict) else {}
    values = selected.get("provenance_source_ids") if isinstance(selected, dict) else []
    return {normalize_string(value) for value in values or [] if normalize_string(value)}


def source_priority(source: dict[str, Any], selected_ids: set[str]) -> tuple[int, str]:
    key = source_key(source)
    source_type = normalize_string(source.get("source_type"))
    if key in selected_ids:
        return (0, key)
    if source_type in {"calendar_event", "gws_calendar_overlap", "gws_calendar_event_detail"}:
        return (1, key)
    return (2, key)


def compact_source(source: dict[str, Any], *, snippet_chars: int) -> dict[str, Any]:
    metadata = source.get("metadata") if isinstance(source.get("metadata"), dict) else {}
    return {
        "source_type": normalize_string(source.get("source_type")),
        "source_id": normalize_string(source.get("source_id")),
        "label": normalize_string(source.get("label")),
        "uri": normalize_string(source.get("uri")),
        "snippet": normalize_string(source.get("snippet"))[:snippet_chars],
        "metadata": {
            key: value
            for key, value in metadata.items()
            if key
            in {
                "profile",
                "model",
                "record_id",
                "message_id",
                "related_model",
                "related_record_id",
                "calendar_id",
                "calendar_summary",
                "event_summary",
                "event_start",
                "event_end",
                "coverage",
                "group_id",
                "query",
                "labels",
            }
        },
    }


def build_supporting_context(
    route: dict[str, Any],
    *,
    max_sources: int,
    snippet_chars: int,
    all_provenance: bool = False,
) -> dict[str, Any]:
    provenance_pack = route.get("provenance_pack") if isinstance(route.get("provenance_pack"), dict) else {}
    sources = provenance_pack.get("sources") if isinstance(provenance_pack.get("sources"), list) else []
    excluded_sources = (
        provenance_pack.get("excluded_sources") if isinstance(provenance_pack.get("excluded_sources"), list) else []
    )
    selected_ids = selected_source_ids(route)
    selected_candidate = route.get("selected_candidate") if isinstance(route.get("selected_candidate"), dict) else {}
    warnings = route.get("warnings") if isinstance(route.get("warnings"), list) else []

    eligible = []
    for source in sources:
        if not isinstance(source, dict):
            continue
        key = source_key(source)
        source_type = normalize_string(source.get("source_type"))
        include = all_provenance or key in selected_ids or source_type in {
            "calendar_event",
            "gws_calendar_overlap",
            "gws_calendar_event_detail",
        }
        if include:
            eligible.append(source)
    eligible = sorted(eligible, key=lambda source: source_priority(source, selected_ids))
    compact_sources = [compact_source(source, snippet_chars=snippet_chars) for source in eligible[:max_sources]]
    return {
        "guidance": (
            "Use these sources as supporting context for the upgraded readout. "
            "Cite source labels or source_type/source_id in evidence fields when they influence conclusions. "
            "Do not treat low-confidence or advisory sources as proof."
        ),
        "route_status": route.get("status"),
        "review_required": route.get("review_required"),
        "selected_candidate": selected_candidate,
        "sources": compact_sources,
        "warnings": [normalize_string(item) for item in warnings if normalize_string(item)],
        "excluded_source_count": len(excluded_sources),
    }


def build_contextual_artifact(
    transcript: dict[str, Any],
    prior_readout: dict[str, Any],
    route: dict[str, Any],
    *,
    max_sources: int,
    snippet_chars: int,
    all_provenance: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    supporting_context = build_supporting_context(
        route,
        max_sources=max_sources,
        snippet_chars=snippet_chars,
        all_provenance=all_provenance,
    )
    artifact = {
        **transcript,
        "prior_readout": {
            key: prior_readout.get(key)
            for key in (
                "title",
                "summary",
                "participants",
                "topics",
                "key_decisions",
                "action_items",
                "unresolved_questions",
                "matter_candidates",
                "memory_candidates",
                "risks",
                "next_steps",
            )
            if key in prior_readout
        },
        "route_decision": {
            "status": route.get("status"),
            "review_required": route.get("review_required"),
            "selected_candidate": route.get("selected_candidate"),
            "confidence_threshold": route.get("confidence_threshold"),
        },
        "supporting_context": supporting_context,
    }
    contextualization = {
        "mode": "contextual_reread",
        "route_status": route.get("status"),
        "review_required": route.get("review_required"),
        "selected_candidate": route.get("selected_candidate"),
        "supporting_context_sources": supporting_context["sources"],
        "warnings": supporting_context["warnings"],
        "excluded_source_count": supporting_context["excluded_source_count"],
    }
    return artifact, contextualization


def call_provider(args: argparse.Namespace, artifact: dict[str, Any], provider_info: dict[str, Any]) -> dict[str, Any]:
    if args.provider == "openai-compatible":
        api_key, api_key_source = summarize_transcript.resolve_openai_key(args)
        if not api_key:
            raise TranscriptionError(
                "openai-compatible contextual rereads require an API key. Provide --openai-api-key, "
                "set OPENAI_API_KEY, or add openai_api_key to api_keys.json."
            )
        base_url, base_url_source = summarize_transcript.resolve_openai_base_url(args)
        provider_info.update(
            {
                "base_url": base_url,
                "api_key_source": api_key_source,
                "base_url_source": base_url_source,
            }
        )
        return summarize_transcript.call_openai_compatible(
            artifact,
            api_key=api_key,
            base_url=base_url,
            model=args.model,
            timeout=args.timeout,
            temperature=args.temperature,
        )

    provider_info["execution"] = "codex exec --sandbox read-only --ask-for-approval never"
    return summarize_transcript.call_codex_exec(artifact, model=args.model, timeout=args.timeout)


def generate_contextual_readout(args: argparse.Namespace) -> tuple[Path, Path]:
    transcript_path = args.transcript.expanduser().resolve()
    readout_path = args.readout.expanduser().resolve()
    route_path = args.route.expanduser().resolve()
    transcript = load_json_object(transcript_path, "transcript artifact")
    prior_readout = load_json_object(readout_path, "readout artifact")
    route = load_json_object(route_path, "route decision")
    artifact, contextualization = build_contextual_artifact(
        transcript,
        prior_readout,
        route,
        max_sources=args.max_sources,
        snippet_chars=args.snippet_chars,
        all_provenance=args.all_provenance,
    )
    provider_info = {
        "name": args.provider,
        "model": args.model,
        "mode": "contextual_reread",
        "prior_readout_path": str(readout_path),
        "route_decision_path": str(route_path),
    }
    model_payload = call_provider(args, artifact, provider_info)
    readout = Readout.from_model_payload(
        model_payload,
        source_artifact_path=transcript_path,
        provider=provider_info,
        contextualization=contextualization,
    )
    base_path = output_base_path(route_path, args.output_dir)
    json_path = Path(str(base_path) + ".readout.json")
    markdown_path = Path(str(base_path) + ".readout.md")
    readout.write_json(json_path)
    readout.write_markdown(markdown_path)
    if args.store:
        ingest_artifact(json_path, root=args.store_dir)
    return json_path, markdown_path


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    try:
        json_path, markdown_path = generate_contextual_readout(args)
    except (TranscriptionError, TranscriptStoreError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Writing contextual readout JSON to {json_path}...")
    print(f"Writing contextual readout Markdown to {markdown_path}...")
    print(f"{CONTEXTUAL_READOUT_JSON_STDOUT_PREFIX}{json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
