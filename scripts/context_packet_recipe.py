#!/usr/bin/env python3
"""
Print downstream commands for a transcript-store compact context JSON packet.
"""
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any, Iterable, Optional, TextIO


class ContextRecipeError(RuntimeError):
    pass


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read compact JSON from transcript_store.py context/search --context and print "
            "shell commands for readout, routing, and contextual reread."
        )
    )
    parser.add_argument(
        "context_json",
        nargs="?",
        type=Path,
        help="Context JSON path. Omit or pass '-' to read stdin.",
    )
    parser.add_argument("--readout", type=Path, help="Existing *.readout.json path to use in route/reread commands.")
    parser.add_argument("--route", type=Path, help="Existing or intended *.route.json path to use in reread command.")
    parser.add_argument("--provider", default="codex-exec", help="Provider for summarize/contextual_reread commands.")
    parser.add_argument("--model", help="Optional model for summarize/contextual_reread commands.")
    parser.add_argument("--store", action="store_true", help="Add --store to readout generation commands.")
    parser.add_argument(
        "--with-provenance",
        action="store_true",
        help="Add read-only gws, Graphiti, and Odollo provenance flags to route_transcript.py.",
    )
    return parser.parse_args(argv)


def load_packet(path: Optional[Path], *, stdin: TextIO = sys.stdin) -> dict[str, Any]:
    if path is None or str(path) == "-":
        raw = stdin.read()
        source = "stdin"
    else:
        source = str(path)
        try:
            raw = path.expanduser().read_text(encoding="utf-8")
        except OSError as exc:
            raise ContextRecipeError(f"Failed to read {path}: {exc}") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ContextRecipeError(f"{source} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ContextRecipeError(f"{source} must contain a JSON object.")
    return payload


def context_payload(packet: dict[str, Any]) -> dict[str, Any]:
    context = packet.get("context") if isinstance(packet.get("context"), dict) else packet
    if not isinstance(context, dict):
        raise ContextRecipeError("Context packet must be an object.")
    return context


def transcript_path_from_context(context: dict[str, Any]) -> str:
    document = context.get("document") if isinstance(context.get("document"), dict) else {}
    path = str(document.get("source_path") or "").strip()
    if not path:
        raise ContextRecipeError("Context packet does not include context.document.source_path.")
    return path


def context_summary(packet: dict[str, Any], context: dict[str, Any]) -> list[str]:
    document = context.get("document") if isinstance(context.get("document"), dict) else {}
    chunk = context.get("chunk") if isinstance(context.get("chunk"), dict) else {}
    media = context.get("media") if isinstance(context.get("media"), dict) else {}
    lines = [
        f"# Transcript: {document.get('title') or document.get('id') or 'unknown'}",
        f"# Document id: {document.get('id') or ''}",
    ]
    if "query" in packet:
        lines.append(f"# Query: {packet.get('query')}")
    if packet.get("selected_rank"):
        lines.append(f"# Selected rank: {packet.get('selected_rank')} of {packet.get('result_count')}")
    if chunk.get("chunk_index") is not None:
        lines.append(f"# Chunk: {chunk.get('chunk_index')}")
    if media.get("start_timestamp"):
        lines.append(f"# Timestamp: {media.get('start_timestamp')} - {media.get('end_timestamp') or ''}".rstrip())
    if media.get("seek_hint"):
        lines.append(f"# Media seek: {media.get('seek_hint')}")
    return lines


def command(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts if str(part))


def build_recipe(packet: dict[str, Any], args: argparse.Namespace) -> str:
    context = context_payload(packet)
    transcript = transcript_path_from_context(context)
    readout = str(args.readout.expanduser()) if args.readout else "<READOUT_JSON_FROM_PREVIOUS_COMMAND>"
    route = str(args.route.expanduser()) if args.route else "<ROUTE_DECISION_JSON_FROM_PREVIOUS_COMMAND>"

    summarize = ["python", "summarize_transcript.py", transcript, "--provider", args.provider]
    reread = ["python", "contextual_reread.py", transcript, readout, route, "--provider", args.provider]
    if args.model:
        summarize.extend(["--model", args.model])
        reread.extend(["--model", args.model])
    if args.store:
        summarize.append("--store")
        reread.append("--store")

    route_command = ["python", "route_transcript.py", transcript, readout]
    if args.with_provenance:
        route_command.extend(["--gws-provenance", "--graphiti-provenance", "--odollo-provenance"])

    lines = [
        *context_summary(packet, context),
        "",
        "# 1. Generate or refresh the first-pass readout.",
        command(summarize),
        "# Capture READOUT_JSON=... from stdout and pass it as --readout next time, or set READOUT below.",
        f"READOUT={shlex.quote(readout)}",
        "",
        "# 2. Create the route decision from the transcript and readout.",
        command(route_command),
        "# Capture ROUTE_DECISION_JSON=... from stdout and pass it as --route next time, or set ROUTE below.",
        f"ROUTE={shlex.quote(route)}",
        "",
        "# 3. Generate the contextual reread from transcript, readout, and route.",
        command(reread),
    ]
    return "\n".join(lines)


def main(argv: Optional[Iterable[str]] = None, *, stdin: TextIO = sys.stdin) -> int:
    args = parse_args(argv)
    try:
        packet = load_packet(args.context_json, stdin=stdin)
        print(build_recipe(packet, args))
    except ContextRecipeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
