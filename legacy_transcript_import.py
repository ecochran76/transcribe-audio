#!/usr/bin/env python3
"""
Convert legacy transcript TXT/DOCX outputs into store-ingestable transcript artifacts.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from docx import Document

from transcript_artifacts import ARTIFACT_SCHEMA_VERSION
from transcript_store import (
    DEFAULT_BACKFILL_EXCLUDE_PATTERNS,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_PROVIDER,
    TranscriptStoreError,
    connect,
    existing_artifact_status,
    ingest_artifact,
    init_db,
    parse_object_json,
    sha256_file,
    stable_id,
    store_dir,
    summarize_backfill_status,
)

LEGACY_IMPORT_JSON_STDOUT_PREFIX = "LEGACY_TRANSCRIPT_IMPORT_JSON="
DEFAULT_LEGACY_PATTERNS = ("*Transcript.txt", "*Transcript.docx", "*transcript.txt", "*transcript.docx")
DEFAULT_MEDIA_EXTENSIONS = (".m4a", ".mp3", ".wav", ".mp4", ".mov", ".mkv", ".webm")


def normalize_string(value: Any) -> str:
    return str(value or "").strip()


def utc_from_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).replace(microsecond=0).isoformat()


def read_legacy_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return path.read_text(encoding="utf-8", errors="replace").strip()
    if suffix == ".docx":
        document = Document(str(path))
        paragraphs = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
        return "\n".join(paragraphs).strip()
    raise TranscriptStoreError(f"Unsupported legacy transcript extension: {path.suffix}")


def transcript_base_name(path: Path) -> str:
    name = path.stem
    return re.sub(r"\s+Transcript$", "", name, flags=re.IGNORECASE).strip() or path.stem


def artifact_title(path: Path) -> str:
    return transcript_base_name(path)


def dedupe_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def legacy_artifact_path(root: Path, source_path: Path) -> Path:
    resolved = source_path.expanduser().resolve()
    item_id = stable_id("legacy-transcript", str(resolved))
    return root / "legacy-artifacts" / item_id[:2] / f"{item_id}-{transcript_base_name(resolved)}.transcript.json"


def media_candidate_paths(roots: Iterable[Path]) -> list[Path]:
    selected: dict[str, Path] = {}
    for root in roots:
        expanded = root.expanduser()
        if expanded.is_file() and expanded.suffix.lower() in DEFAULT_MEDIA_EXTENSIONS:
            try:
                selected[str(expanded.resolve())] = expanded.resolve()
            except OSError:
                pass
            continue
        if not expanded.exists() or not expanded.is_dir():
            continue
        for path in expanded.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in DEFAULT_MEDIA_EXTENSIONS:
                continue
            try:
                selected[str(path.resolve())] = path.resolve()
            except OSError:
                continue
    return [selected[key] for key in sorted(selected)]


def read_media_index(path: Path) -> list[Path]:
    try:
        lines = path.expanduser().read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        raise TranscriptStoreError(f"Failed to read media index {path}: {exc}") from exc
    selected: dict[str, Path] = {}
    for line in lines:
        value = normalize_string(line)
        if not value:
            continue
        media_path = Path(value).expanduser()
        if media_path.exists() and media_path.is_file() and media_path.suffix.lower() in DEFAULT_MEDIA_EXTENSIONS:
            try:
                selected[str(media_path.resolve())] = media_path.resolve()
            except OSError:
                continue
    return [selected[key] for key in sorted(selected)]


def find_matching_media(source_path: Path, media_paths: Iterable[Path]) -> str:
    base = transcript_base_name(source_path).lower()
    for candidate in media_paths:
        candidate_base = candidate.stem.lower()
        if candidate_base == base or base in candidate_base or candidate_base in base:
            return str(candidate.resolve())
    return ""


def legacy_candidate_paths(
    roots: Iterable[Path],
    *,
    recursive: bool = False,
    patterns: Iterable[str] = DEFAULT_LEGACY_PATTERNS,
    exclude_patterns: Iterable[str] = DEFAULT_BACKFILL_EXCLUDE_PATTERNS,
    modified_within_days: Optional[float] = None,
) -> list[Path]:
    import time

    min_mtime = None
    if modified_within_days is not None:
        min_mtime = time.time() - (modified_within_days * 86400.0)
    normalized_patterns = tuple(patterns) or DEFAULT_LEGACY_PATTERNS
    normalized_excludes = tuple(exclude_patterns)
    selected: dict[str, Path] = {}
    for root in roots:
        expanded = root.expanduser()
        if expanded.is_file():
            paths = [expanded]
        elif expanded.is_dir():
            iterator = expanded.rglob("*") if recursive else expanded.glob("*")
            paths = [path for path in iterator if path.is_file()]
        else:
            continue
        for path in paths:
            path_text = str(path)
            if any(fnmatch.fnmatch(path_text, pattern) for pattern in normalized_excludes):
                continue
            if not any(fnmatch.fnmatch(path.name, pattern) for pattern in normalized_patterns):
                continue
            try:
                if min_mtime is not None and path.stat().st_mtime < min_mtime:
                    continue
                resolved = path.resolve()
            except OSError:
                continue
            selected[str(resolved)] = resolved
    return [selected[key] for key in sorted(selected)]


def legacy_artifact_payload(source_path: Path, *, media_path: str = "") -> dict[str, Any]:
    text = read_legacy_text(source_path)
    timestamp = utc_from_mtime(source_path)
    output_key = "docx" if source_path.suffix.lower() == ".docx" else "txt"
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "source_media_path": media_path,
        "working_media_path": media_path,
        "backend": "legacy-import",
        "duration_seconds": 0.0,
        "recording_start": timestamp,
        "recording_end": timestamp,
        "transcript_window_start_seconds": 0.0,
        "transcript_window_end_seconds": 0.0,
        "utterance_count": 0,
        "transcript_text": text,
        "utterances": [],
        "output_paths": {
            output_key: str(source_path),
        },
        "event": None,
        "transcript_title": artifact_title(source_path),
        "legacy_import": {
            "source_path": str(source_path),
            "source_sha256": sha256_file(source_path),
            "needs_enrichment": True,
            "notes": "Synthesized from a legacy transcript output without a modern sidecar.",
        },
    }


def plan_legacy_import(
    roots: Iterable[Path],
    *,
    root: Optional[Path] = None,
    recursive: bool = False,
    patterns: Iterable[str] = DEFAULT_LEGACY_PATTERNS,
    exclude_patterns: Iterable[str] = DEFAULT_BACKFILL_EXCLUDE_PATTERNS,
    modified_within_days: Optional[float] = None,
    media_roots: Iterable[Path] = (),
    media_index_files: Iterable[Path] = (),
    no_media_match: bool = False,
    limit: Optional[int] = None,
    force: bool = False,
    dedupe: bool = True,
    embedding_provider: str = DEFAULT_EMBEDDING_PROVIDER,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> list[dict[str, Any]]:
    runtime_root = store_dir(root)
    indexed_media_paths: list[Path] = []
    if not no_media_match:
        for media_index_file in media_index_files:
            indexed_media_paths.extend(read_media_index(media_index_file))
    fallback_media_roots: tuple[Path, ...] = ()
    if not no_media_match and (media_roots or not indexed_media_paths):
        fallback_media_roots = (*media_roots, *(candidate.parent for candidate in roots if candidate.expanduser().is_file()))
    media_paths = [*indexed_media_paths, *media_candidate_paths(fallback_media_roots)]
    existing_hashes, existing_titles = existing_legacy_dedupe_keys(runtime_root)
    seen_hashes: set[str] = set()
    seen_titles: set[str] = set()
    candidates = legacy_candidate_paths(
        roots,
        recursive=recursive,
        patterns=patterns,
        exclude_patterns=exclude_patterns,
        modified_within_days=modified_within_days,
    )
    items: list[dict[str, Any]] = []
    for candidate in candidates:
        artifact_path = legacy_artifact_path(runtime_root, candidate)
        source_hash = ""
        try:
            source_hash = sha256_file(candidate)
        except OSError:
            pass
        title = artifact_title(candidate)
        title_key = dedupe_key(title)
        media_path = find_matching_media(candidate, media_paths)
        status = "convert"
        reason = ""
        if artifact_path.exists():
            try:
                existing = existing_artifact_status(
                    artifact_path,
                    root=runtime_root,
                    embedding_provider=embedding_provider,
                    embedding_model=embedding_model,
                )
                status = existing["status"]
            except TranscriptStoreError as exc:
                status = "error"
                error = str(exc)
            else:
                error = ""
        else:
            error = ""
        if dedupe and source_hash and source_hash in existing_hashes and not force:
            status = "duplicate_existing"
            reason = "Existing imported legacy transcript has the same source hash."
        elif dedupe and title_key and title_key in existing_titles and not force:
            status = "duplicate_existing"
            reason = "Existing stored transcript has the same normalized title."
        elif dedupe and source_hash and source_hash in seen_hashes and not force:
            status = "duplicate_in_batch"
            reason = "Earlier candidate in this batch has the same source hash."
        elif dedupe and title_key and title_key in seen_titles and not force:
            status = "duplicate_in_batch"
            reason = "Earlier candidate in this batch has the same normalized title."
        if force and status == "skip":
            status = "update"
        if source_hash:
            seen_hashes.add(source_hash)
        if title_key:
            seen_titles.add(title_key)
        item = {
            "kind": "legacy_transcript",
            "title": title,
            "source_path": str(candidate),
            "source_sha256": source_hash,
            "artifact_path": str(artifact_path),
            "media_path": media_path,
            "status": status,
        }
        if reason:
            item["reason"] = reason
        if error:
            item["error"] = error
        items.append(item)
        if limit is not None and len(items) >= limit:
            break
    return items


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def existing_legacy_dedupe_keys(root: Path) -> tuple[set[str], set[str]]:
    hashes: set[str] = set()
    titles: set[str] = set()
    with connect(root) as con:
        init_db(con)
        rows = con.execute("SELECT title, json_payload FROM documents WHERE kind = 'transcript'").fetchall()
    for row in rows:
        title_key = dedupe_key(str(row["title"] or ""))
        if title_key:
            titles.add(title_key)
        payload = parse_object_json(row["json_payload"])
        legacy = payload.get("legacy_import") if isinstance(payload.get("legacy_import"), dict) else {}
        source_hash = normalize_string(legacy.get("source_sha256"))
        if source_hash:
            hashes.add(source_hash)
    return hashes, titles


def apply_legacy_import(
    planned: list[dict[str, Any]],
    *,
    root: Optional[Path] = None,
    force: bool = False,
    embedding_provider: str = DEFAULT_EMBEDDING_PROVIDER,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> list[dict[str, Any]]:
    runtime_root = store_dir(root)
    results: list[dict[str, Any]] = []
    for item in planned:
        if item["status"] == "error":
            results.append({**item, "result": "error"})
            continue
        if str(item["status"]).startswith("duplicate_"):
            results.append({**item, "result": "skipped"})
            continue
        artifact_path = Path(item["artifact_path"])
        if item["status"] == "skip" and not force:
            results.append({**item, "result": "skipped"})
            continue
        try:
            payload = legacy_artifact_payload(Path(item["source_path"]), media_path=normalize_string(item.get("media_path")))
            write_json(artifact_path, payload)
            ingest_result = ingest_artifact(
                artifact_path,
                root=runtime_root,
                embedding_provider=embedding_provider,
                embedding_model=embedding_model,
            ).to_dict()
        except Exception as exc:  # noqa: BLE001 - keep batch import moving and report per-item failure.
            results.append({**item, "result": "error", "error": str(exc)})
            continue
        results.append({**item, "result": ingest_result["status"], "stored_path": ingest_result["stored_path"]})
    return results


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import legacy TXT/DOCX transcript outputs into ~/.transcripts.")
    parser.add_argument("roots", nargs="+", type=Path, help="Files or directories to scan.")
    parser.add_argument("--store-dir", type=Path, default=None, help="Store root. Defaults to ~/.transcripts.")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--pattern", action="append", dest="patterns", help="Transcript filename glob. Repeatable.")
    parser.add_argument("--exclude", action="append", dest="exclude_patterns", help="Path glob to exclude. Repeatable.")
    parser.add_argument("--modified-within-days", type=float)
    parser.add_argument("--media-root", action="append", dest="media_roots", type=Path, help="Directory to search for matching source media. Repeatable.")
    parser.add_argument(
        "--media-index-file",
        action="append",
        dest="media_index_files",
        type=Path,
        help="Newline-delimited media path index. Repeatable. Avoids slow Python directory walks on mounted drives.",
    )
    parser.add_argument("--no-media-match", action="store_true", help="Skip source recording matching for this import.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--apply", action="store_true", help="Write synthesized sidecars and ingest them.")
    parser.add_argument("--force", action="store_true", help="Rebuild sidecars and re-ingest current rows.")
    parser.add_argument("--no-dedupe", action="store_true", help="Do not skip same-hash or same-title legacy candidates.")
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
    planned = plan_legacy_import(
        args.roots,
        root=runtime_root,
        recursive=args.recursive,
        patterns=args.patterns or DEFAULT_LEGACY_PATTERNS,
        exclude_patterns=(*DEFAULT_BACKFILL_EXCLUDE_PATTERNS, *(args.exclude_patterns or [])),
        modified_within_days=args.modified_within_days,
        media_roots=args.media_roots or [],
        media_index_files=args.media_index_files or [],
        no_media_match=args.no_media_match,
        limit=args.limit,
        force=args.force,
        dedupe=not args.no_dedupe,
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
    )
    items = (
        apply_legacy_import(
            planned,
            root=runtime_root,
            force=args.force,
            embedding_provider=args.embedding_provider,
            embedding_model=args.embedding_model,
        )
        if args.apply
        else planned
    )
    payload = {
        "store_dir": str(runtime_root),
        "dry_run": not args.apply,
        "selected_count": len(planned),
        **summarize_backfill_status(items),
        "items": items,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"{LEGACY_IMPORT_JSON_STDOUT_PREFIX}{runtime_root / 'legacy-artifacts'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
