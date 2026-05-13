#!/usr/bin/env python3
"""
Apply local-filesystem deposition actions from a deposition preview artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

from deposition_artifacts import AppliedDepositAction, AppliedDepositFile, DepositApplyResult, normalize_string
from transcribe_common import TranscriptionError

DEPOSITION_APPLY_JSON_STDOUT_PREFIX = "DEPOSITION_APPLY_JSON="


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply local filesystem actions from a deposition preview JSON.")
    parser.add_argument("preview", type=Path, help="Path to a *.deposit-preview.json artifact.")
    parser.add_argument("--output-dir", type=Path, help="Directory for apply result output. Defaults beside preview.")
    parser.add_argument(
        "--allow-review-required",
        action="store_true",
        help="Allow local apply even when the preview route required review.",
    )
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
    return directory / f"{base_name}.deposit-apply.json"


def slugify(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip()).strip("-").lower()
    return normalized[:80] or "unrouted"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def versioned_destination(path: Path, source_hash: str) -> tuple[Path, str]:
    if not path.exists():
        return path, "copied"
    if path.is_file() and sha256_file(path) == source_hash:
        return path, "skipped_existing_same_hash"
    suffixes = path.suffixes
    if suffixes:
        suffix = "".join(suffixes)
        stem = path.name[: -len(suffix)]
    else:
        suffix = ""
        stem = path.name
    for index in range(2, 1000):
        candidate = path.with_name(f"{stem}.v{index}{suffix}")
        if not candidate.exists():
            return candidate, "copied_versioned"
        if candidate.is_file() and sha256_file(candidate) == source_hash:
            return candidate, "skipped_existing_same_hash"
    raise TranscriptionError(f"Could not find an available versioned destination for {path}.")


def selected_slug(preview: dict[str, Any]) -> str:
    selected = preview.get("selected_candidate") if isinstance(preview.get("selected_candidate"), dict) else {}
    label = normalize_string(selected.get("label"))
    target_id = normalize_string(selected.get("target_id"))
    return slugify(label or target_id or "unrouted")


def apply_local_action(action: dict[str, Any], *, preview: dict[str, Any]) -> AppliedDepositAction:
    target_root = Path(normalize_string(action.get("target_id"))).expanduser()
    if not target_root:
        raise TranscriptionError("Local filesystem deposition action is missing target_id.")
    target_dir = target_root / selected_slug(preview)
    target_dir.mkdir(parents=True, exist_ok=True)
    applied_files: list[AppliedDepositFile] = []
    for source_value in action.get("source_paths") or []:
        source_path = Path(str(source_value)).expanduser()
        if not source_path.exists() or not source_path.is_file():
            raise TranscriptionError(f"Deposition source file does not exist: {source_path}")
        source_hash = sha256_file(source_path)
        destination, status = versioned_destination(target_dir / source_path.name, source_hash)
        if status.startswith("copied"):
            shutil.copy2(source_path, destination)
        applied_files.append(
            AppliedDepositFile(
                source_path=str(source_path),
                destination_path=str(destination),
                status=status,
                sha256=source_hash,
                bytes=source_path.stat().st_size,
            )
        )
    action_status = "applied" if any(file.status.startswith("copied") for file in applied_files) else "skipped"
    return AppliedDepositAction(
        action_type=normalize_string(action.get("action_type")) or "copy_artifacts",
        target_kind="local_filesystem",
        target_id=str(target_root),
        target_profile=normalize_string(action.get("target_profile")),
        status=action_status,
        files=applied_files,
        metadata={"selected_slug": target_dir.name},
    )


def apply_preview(args: argparse.Namespace) -> Path:
    preview_path = args.preview.expanduser().resolve()
    preview = load_preview(preview_path)
    if preview.get("review_required") and not args.allow_review_required:
        raise TranscriptionError(
            "Refusing to apply a deposition preview that requires review. Pass --allow-review-required to override."
        )
    applied_actions: list[AppliedDepositAction] = []
    for action in preview.get("actions") or []:
        if not isinstance(action, dict):
            continue
        target_kind = normalize_string(action.get("target_kind"))
        if target_kind == "local_filesystem":
            applied_actions.append(apply_local_action(action, preview=preview))
            continue
        applied_actions.append(
            AppliedDepositAction(
                action_type=normalize_string(action.get("action_type")),
                target_kind=target_kind,
                target_id=normalize_string(action.get("target_id")),
                target_profile=normalize_string(action.get("target_profile")),
                status="skipped",
                reason="Only local_filesystem deposition apply is implemented.",
                metadata={"writes_enabled": False},
            )
        )
    result = DepositApplyResult(source_preview_path=str(preview_path), actions=applied_actions)
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
    print(f"Writing deposition apply JSON to {path}...")
    print(f"{DEPOSITION_APPLY_JSON_STDOUT_PREFIX}{path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
