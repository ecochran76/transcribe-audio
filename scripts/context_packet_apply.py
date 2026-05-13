#!/usr/bin/env python3
"""
Preview or execute downstream steps for a transcript-store context packet.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, TextIO

import context_packet_recipe

READOUT_PREFIX = "READOUT_JSON="
ROUTE_PREFIX = "ROUTE_DECISION_JSON="
CONTEXTUAL_PREFIX = "CONTEXTUAL_READOUT_JSON="
DEFAULT_MANIFEST_DIR = Path("~/.local/state/transcribe-audio/context-packet-runs")
REPO_ROOT = Path(__file__).resolve().parents[1]


class ContextApplyError(RuntimeError):
    pass


Runner = Callable[..., subprocess.CompletedProcess[str]]


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview or execute readout, route, and contextual-reread steps for a compact context packet."
    )
    parser.add_argument(
        "context_json",
        nargs="?",
        type=Path,
        help="Context JSON path. Omit or pass '-' to read stdin.",
    )
    parser.add_argument("--apply", action="store_true", help="Execute the planned commands. Default is preview only.")
    parser.add_argument("--readout", type=Path, help="Existing *.readout.json path; skips readout generation.")
    parser.add_argument("--route", type=Path, help="Existing *.route.json path; skips route generation.")
    parser.add_argument("--provider", default="codex-exec", help="Provider for summarize/contextual_reread commands.")
    parser.add_argument("--model", help="Optional model for summarize/contextual_reread commands.")
    parser.add_argument(
        "--python",
        dest="python_executable",
        type=Path,
        help="Python executable for child commands. Defaults to repo .venv/bin/python when present.",
    )
    parser.add_argument("--store", action="store_true", help="Add --store to generated readout commands.")
    parser.add_argument(
        "--with-provenance",
        action="store_true",
        help="Add read-only gws, Graphiti, and Odollo provenance flags to route_transcript.py.",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=DEFAULT_MANIFEST_DIR,
        help="Directory for executed-run manifests. Used only with --apply.",
    )
    parser.add_argument("--no-manifest", action="store_true", help="Do not write an executed-run manifest.")
    parser.add_argument("--list-manifests", action="store_true", help="List recent executed-run manifests and exit.")
    parser.add_argument("--limit", type=int, default=10, help="Maximum manifests to list with --list-manifests.")
    parser.add_argument("--timeout", type=float, default=300.0, help="Timeout per executed command.")
    parser.add_argument(
        "--provider-timeout",
        type=float,
        help="Provider request timeout passed to summarize/contextual_reread child commands.",
    )
    parser.add_argument("--format", choices=("text", "json"), default="text", help="Output format.")
    return parser.parse_args(argv)


def child_python(args: argparse.Namespace) -> str:
    if args.python_executable:
        return str(args.python_executable.expanduser())
    venv_python = REPO_ROOT / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def extract_prefixed_path(stdout: str, prefix: str) -> str:
    values = [line[len(prefix) :].strip() for line in stdout.splitlines() if line.startswith(prefix)]
    if not values:
        raise ContextApplyError(f"Command output did not include {prefix}<path>.")
    return values[-1]


def command_record(name: str, argv: list[str], *, status: str = "pending", path: str = "") -> dict[str, Any]:
    return {"name": name, "argv": argv, "status": status, "path": path}


def build_plan(packet: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    context = context_packet_recipe.context_payload(packet)
    transcript = context_packet_recipe.transcript_path_from_context(context)
    readout = str(args.readout.expanduser()) if args.readout else ""
    route = str(args.route.expanduser()) if args.route else ""
    python_executable = child_python(args)

    summarize = [python_executable, "summarize_transcript.py", transcript, "--provider", args.provider]
    contextual = [
        python_executable,
        "contextual_reread.py",
        transcript,
        readout or "<READOUT_JSON>",
        route or "<ROUTE_JSON>",
        "--provider",
        args.provider,
    ]
    if args.model:
        summarize.extend(["--model", args.model])
        contextual.extend(["--model", args.model])
    if args.provider_timeout is not None:
        summarize.extend(["--timeout", str(args.provider_timeout)])
        contextual.extend(["--timeout", str(args.provider_timeout)])
    if args.store:
        summarize.append("--store")
        contextual.append("--store")

    route_command = [python_executable, "route_transcript.py", transcript, readout or "<READOUT_JSON>"]
    if args.with_provenance:
        route_command.extend(["--gws-provenance", "--graphiti-provenance", "--odollo-provenance"])

    steps = []
    if readout:
        steps.append(command_record("readout", summarize, status="skipped_existing", path=readout))
    else:
        steps.append(command_record("readout", summarize))
    if route:
        steps.append(command_record("route", route_command, status="skipped_existing", path=route))
    else:
        steps.append(command_record("route", route_command))
    steps.append(command_record("contextual_reread", contextual))

    return {
        "mode": "apply" if args.apply else "preview",
        "transcript": transcript,
        "query": packet.get("query"),
        "selected_rank": packet.get("selected_rank"),
        "document_id": (context.get("document") or {}).get("id") if isinstance(context.get("document"), dict) else "",
        "chunk_index": (context.get("chunk") or {}).get("chunk_index") if isinstance(context.get("chunk"), dict) else None,
        "steps": steps,
    }


def run_step(argv: list[str], *, runner: Runner, timeout: float) -> subprocess.CompletedProcess[str]:
    return runner(argv, text=True, capture_output=True, timeout=timeout)


def execute_plan(plan: dict[str, Any], *, runner: Runner = subprocess.run, timeout: float) -> dict[str, Any]:
    readout_path = ""
    route_path = ""
    contextual_path = ""
    for step in plan["steps"]:
        name = step["name"]
        if step["status"] == "skipped_existing":
            if name == "readout":
                readout_path = step["path"]
            elif name == "route":
                route_path = step["path"]
            continue
        argv = list(step["argv"])
        if name == "route":
            argv = [readout_path if item == "<READOUT_JSON>" else item for item in argv]
        elif name == "contextual_reread":
            argv = [
                readout_path if item == "<READOUT_JSON>" else route_path if item == "<ROUTE_JSON>" else item
                for item in argv
            ]
        step["argv"] = argv
        completed = run_step(argv, runner=runner, timeout=timeout)
        step["returncode"] = completed.returncode
        step["stdout"] = completed.stdout
        step["stderr"] = completed.stderr
        if completed.returncode != 0:
            step["status"] = "failed"
            raise ContextApplyError(f"{name} failed with exit code {completed.returncode}.")
        if name == "readout":
            readout_path = extract_prefixed_path(completed.stdout, READOUT_PREFIX)
            step["path"] = readout_path
        elif name == "route":
            route_path = extract_prefixed_path(completed.stdout, ROUTE_PREFIX)
            step["path"] = route_path
        elif name == "contextual_reread":
            contextual_path = extract_prefixed_path(completed.stdout, CONTEXTUAL_PREFIX)
            step["path"] = contextual_path
        step["status"] = "completed"
    plan["readout"] = readout_path
    plan["route"] = route_path
    plan["contextual_readout"] = contextual_path
    plan["status"] = "completed"
    return plan


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def run_id_for_plan(plan: dict[str, Any], created_at: str) -> str:
    seed = "\n".join(
        [
            created_at,
            str(plan.get("transcript") or ""),
            str(plan.get("document_id") or ""),
            str(plan.get("chunk_index") or ""),
            str(plan.get("readout") or ""),
            str(plan.get("route") or ""),
            str(plan.get("contextual_readout") or ""),
        ]
    )
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:20]


def manifest_step(step: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": step.get("name"),
        "status": step.get("status"),
        "argv": step.get("argv") or [],
        "path": step.get("path") or "",
        "returncode": step.get("returncode"),
    }


def build_manifest(plan: dict[str, Any], *, created_at: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": run_id_for_plan(plan, created_at),
        "created_at": created_at,
        "status": plan.get("status"),
        "transcript": plan.get("transcript"),
        "query": plan.get("query"),
        "selected_rank": plan.get("selected_rank"),
        "document_id": plan.get("document_id"),
        "chunk_index": plan.get("chunk_index"),
        "artifacts": {
            "readout": plan.get("readout") or "",
            "route": plan.get("route") or "",
            "contextual_readout": plan.get("contextual_readout") or "",
        },
        "steps": [manifest_step(step) for step in plan.get("steps") or []],
    }


def write_manifest(plan: dict[str, Any], manifest_dir: Path) -> Path:
    manifest = build_manifest(plan, created_at=utc_now())
    root = manifest_dir.expanduser()
    root.mkdir(parents=True, exist_ok=True)
    timestamp = str(manifest["created_at"]).replace(":", "-")
    path = root / f"{timestamp}-{manifest['run_id']}.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return path


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContextApplyError(f"Failed to read manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ContextApplyError(f"Manifest {path} must contain a JSON object.")
    return payload


def manifest_summary(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
    return {
        "path": str(path),
        "run_id": payload.get("run_id") or "",
        "created_at": payload.get("created_at") or "",
        "status": payload.get("status") or "",
        "transcript": payload.get("transcript") or "",
        "query": payload.get("query") or "",
        "document_id": payload.get("document_id") or "",
        "chunk_index": payload.get("chunk_index"),
        "readout": artifacts.get("readout") or "",
        "route": artifacts.get("route") or "",
        "contextual_readout": artifacts.get("contextual_readout") or "",
    }


def list_manifests(manifest_dir: Path, *, limit: int) -> dict[str, Any]:
    root = manifest_dir.expanduser()
    if not root.exists():
        return {"manifest_dir": str(root), "count": 0, "manifests": []}
    items = []
    for path in sorted(root.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        items.append(manifest_summary(path, load_manifest(path)))
        if len(items) >= max(limit, 0):
            break
    return {"manifest_dir": str(root), "count": len(items), "manifests": items}


def format_manifest_list(payload: dict[str, Any]) -> str:
    lines = [f"Manifest dir: {payload['manifest_dir']}", f"Count: {payload['count']}"]
    for item in payload["manifests"]:
        heading = f"{item['created_at']} {item['run_id']} [{item['status']}]"
        lines.extend(
            [
                "",
                heading.strip(),
                f"  transcript: {item['transcript']}",
                f"  document: {item['document_id']} chunk={item['chunk_index']}",
                f"  query: {item['query']}",
                f"  contextual_readout: {item['contextual_readout']}",
                f"  manifest: {item['path']}",
            ]
        )
    return "\n".join(lines)


def format_text(plan: dict[str, Any]) -> str:
    lines = [
        f"Mode: {plan['mode']}",
        f"Transcript: {plan['transcript']}",
    ]
    if plan.get("query"):
        lines.append(f"Query: {plan['query']}")
    if plan.get("document_id"):
        lines.append(f"Document: {plan['document_id']}")
    if plan.get("chunk_index") is not None:
        lines.append(f"Chunk: {plan['chunk_index']}")
    lines.append("")
    for index, step in enumerate(plan["steps"], 1):
        lines.append(f"{index}. {step['name']} [{step['status']}]")
        lines.append(context_packet_recipe.command(step["argv"]))
        if step.get("path"):
            lines.append(f"   path: {step['path']}")
    if plan.get("manifest_path"):
        lines.extend(["", f"Manifest: {plan['manifest_path']}"])
    return "\n".join(lines)


def main(
    argv: Optional[Iterable[str]] = None,
    *,
    stdin: TextIO = sys.stdin,
    runner: Runner = subprocess.run,
) -> int:
    args = parse_args(argv)
    try:
        if args.list_manifests:
            payload = list_manifests(args.manifest_dir, limit=args.limit)
            if args.format == "json":
                print(json.dumps(payload, indent=2, ensure_ascii=False))
            else:
                print(format_manifest_list(payload))
            return 0
        packet = context_packet_recipe.load_packet(args.context_json, stdin=stdin)
        plan = build_plan(packet, args)
        if args.apply:
            plan = execute_plan(plan, runner=runner, timeout=args.timeout)
            if not args.no_manifest:
                plan["manifest_path"] = str(write_manifest(plan, args.manifest_dir))
        if args.format == "json":
            print(json.dumps(plan, indent=2, ensure_ascii=False))
        else:
            print(format_text(plan))
    except (ContextApplyError, context_packet_recipe.ContextRecipeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
