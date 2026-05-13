#!/usr/bin/env python3
"""
Link stored legacy transcript imports to source recording blobs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

from legacy_transcript_import import find_matching_media, media_candidate_paths, read_media_index
from transcript_store import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_PROVIDER,
    TranscriptStoreError,
    connect,
    ingest_artifact,
    init_db,
    parse_object_json,
    store_dir,
)

LEGACY_MEDIA_LINK_JSON_STDOUT_PREFIX = "LEGACY_MEDIA_LINK_JSON="


def normalize_string(value: Any) -> str:
    return str(value or "").strip()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def unlinked_legacy_transcripts(root: Path) -> list[dict[str, Any]]:
    with connect(root) as con:
        init_db(con)
        rows = con.execute(
            """
            SELECT documents.*
            FROM documents
            LEFT JOIN document_blobs
              ON document_blobs.document_id = documents.id
             AND document_blobs.role = 'source_recording'
            WHERE documents.kind = 'transcript'
              AND document_blobs.blob_id IS NULL
            ORDER BY documents.generated_at, documents.title
            """
        ).fetchall()

    items: list[dict[str, Any]] = []
    for row in rows:
        payload = parse_object_json(row["json_payload"])
        legacy = payload.get("legacy_import") if isinstance(payload.get("legacy_import"), dict) else {}
        if legacy.get("needs_enrichment") is not True:
            continue
        items.append(
            {
                "id": row["id"],
                "title": row["title"],
                "source_path": row["source_path"],
                "stored_path": row["stored_path"],
                "legacy_source_path": normalize_string(legacy.get("source_path")),
                "current_media_path": normalize_string(payload.get("source_media_path")),
            }
        )
    return items


def indexed_media_paths(
    *,
    media_index_files: Iterable[Path] = (),
    media_roots: Iterable[Path] = (),
) -> list[Path]:
    selected: dict[str, Path] = {}
    for media_index_file in media_index_files:
        for path in read_media_index(media_index_file):
            selected[str(path)] = path
    for path in media_candidate_paths(media_roots):
        selected[str(path)] = path
    return [selected[key] for key in sorted(selected)]


def plan_media_links(
    *,
    root: Optional[Path] = None,
    media_index_files: Iterable[Path] = (),
    media_roots: Iterable[Path] = (),
    limit: Optional[int] = None,
) -> list[dict[str, Any]]:
    runtime_root = store_dir(root)
    media_paths = indexed_media_paths(media_index_files=media_index_files, media_roots=media_roots)
    items: list[dict[str, Any]] = []
    for item in unlinked_legacy_transcripts(runtime_root):
        match_basis = Path(item["legacy_source_path"] or item["source_path"])
        media_path = find_matching_media(match_basis, media_paths)
        status = "link" if media_path else "no_match"
        items.append({**item, "media_path": media_path, "status": status})
        if limit is not None and len(items) >= limit:
            break
    return items


def apply_media_links(
    planned: list[dict[str, Any]],
    *,
    root: Optional[Path] = None,
    embedding_provider: str = DEFAULT_EMBEDDING_PROVIDER,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> list[dict[str, Any]]:
    runtime_root = store_dir(root)
    results: list[dict[str, Any]] = []
    for item in planned:
        if item["status"] != "link":
            results.append({**item, "result": "skipped"})
            continue
        artifact_path = Path(item["source_path"]).expanduser()
        try:
            payload = parse_object_json(artifact_path.read_text(encoding="utf-8"))
            if not payload:
                raise TranscriptStoreError(f"Artifact {artifact_path} did not contain a JSON object.")
            payload["source_media_path"] = item["media_path"]
            payload["working_media_path"] = item["media_path"]
            write_json(artifact_path, payload)
            ingest_result = ingest_artifact(
                artifact_path,
                root=runtime_root,
                embedding_provider=embedding_provider,
                embedding_model=embedding_model,
            ).to_dict()
        except Exception as exc:  # noqa: BLE001 - report per-item failure and keep batch moving.
            results.append({**item, "result": "error", "error": str(exc)})
            continue
        results.append({**item, "result": ingest_result["status"], "stored_path": ingest_result["stored_path"]})
    return results


def summarize(items: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        value = normalize_string(item.get(key)) or "unknown"
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Link legacy transcript imports to source media blobs.")
    parser.add_argument("--store-dir", type=Path, default=None, help="Store root. Defaults to ~/.transcripts.")
    parser.add_argument(
        "--media-index-file",
        action="append",
        dest="media_index_files",
        type=Path,
        help="Newline-delimited media path index. Repeatable.",
    )
    parser.add_argument(
        "--media-root",
        action="append",
        dest="media_roots",
        type=Path,
        help="Directory to recursively scan for media. Prefer indexes on mounted drives.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--apply", action="store_true", help="Update sidecars and ingest blobs.")
    parser.add_argument(
        "--embedding-provider",
        choices=("ollama", "openai-compatible", "debug-hash", "hash"),
        default=DEFAULT_EMBEDDING_PROVIDER,
    )
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    runtime_root = store_dir(args.store_dir)
    try:
        planned = plan_media_links(
            root=runtime_root,
            media_index_files=args.media_index_files or [],
            media_roots=args.media_roots or [],
            limit=args.limit,
        )
        items = (
            apply_media_links(
                planned,
                root=runtime_root,
                embedding_provider=args.embedding_provider,
                embedding_model=args.embedding_model,
            )
            if args.apply
            else planned
        )
    except TranscriptStoreError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    payload = {
        "store_dir": str(runtime_root),
        "dry_run": not args.apply,
        "selected_count": len(planned),
        "by_status": summarize(planned, "status"),
        "by_result": summarize(items, "result") if args.apply else {},
        "items": items,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"{LEGACY_MEDIA_LINK_JSON_STDOUT_PREFIX}{runtime_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
