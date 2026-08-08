from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import struct
import subprocess
import wave
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from acoustic_audio_derivatives import (
    ensure_private_tree,
    read_private_object,
    require_private_file,
    sha256_file,
    write_immutable_private_json,
)
from acoustic_plan0057_review import CHRIS_SUBJECT_ID, ERIC_SUBJECT_ID
from acoustic_review_surface import render_review_surface
from acoustic_shadow_evidence import canonical_hash


PREVIEW_SCHEMA = "transcribe-audio.plan0058-review-surface-preview.v1"
MANIFEST_SCHEMA = "transcribe-audio.plan0058-review-surface-manifest.v1"
RECEIPT_SCHEMA = "transcribe-audio.plan0058-review-surface-receipt.v1"
REPLAY_SCHEMA = "transcribe-audio.plan0058-review-surface-replay.v1"
MODULE_PATH = Path(__file__).name
DEFAULT_RUNTIME_ROOT = Path("~/.local/state/transcribe-audio/plan-0058")
EXPECTED_CARD_COUNT = 15
ALLOWED_SUBJECT_IDS = frozenset({CHRIS_SUBJECT_ID, ERIC_SUBJECT_ID})
NEGATIVE_ACTIONS = {
    "apply_speaker_assignments": False,
    "create_or_mutate_identities": False,
    "create_or_mutate_contacts": False,
    "create_or_mutate_relationships": False,
    "mutate_profiles_or_references": False,
    "write_external_provider": False,
    "write_graphiti": False,
    "enable_default_integration": False,
    "run_historical_reprocessing": False,
    "run_fresh_acoustic_cohort": False,
    "modify_previews_runtime": False,
}


class Plan0058Error(ValueError):
    """Raised when the bounded review-surface proof is not trustworthy."""


def _git(arguments: Sequence[str], *, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", *arguments],
        cwd=Path(__file__).resolve().parent,
        capture_output=True,
        text=not binary,
        check=False,
    )
    if result.returncode:
        raise Plan0058Error("Repository authority is unavailable.")
    return result.stdout if binary else result.stdout.strip()


def _repository_authority() -> dict[str, Any]:
    if _git(["status", "--porcelain=v1", "--untracked-files=normal"]):
        raise Plan0058Error("Repository must be clean before fixture apply.")
    if str(
        _git(["rev-list", "--left-right", "--count", "HEAD...@{upstream}"])
    ).split() != ["0", "0"]:
        raise Plan0058Error("Repository must be upstream-even before fixture apply.")
    commit = str(_git(["rev-parse", "HEAD"]))
    body = _git(["show", f"{commit}:{MODULE_PATH}"], binary=True)
    if (
        not isinstance(body, bytes)
        or hashlib.sha256(body).hexdigest() != sha256_file(Path(__file__).resolve())
    ):
        raise Plan0058Error("Committed Plan 0058 authority drifted.")
    return {
        "commit": commit,
        "module_sha256": hashlib.sha256(body).hexdigest(),
        "clean": True,
        "upstream_ahead": 0,
        "upstream_behind": 0,
    }


def _card_shape() -> list[tuple[str, int]]:
    return [("synthetic-a", 3), ("synthetic-b", 6), ("synthetic-c", 6)]


def _wav_bytes(ordinal: int) -> bytes:
    sample_rate = 16_000
    duration_seconds = 1.25
    frequency = 220.0 + ordinal * 17.0
    frame_count = int(sample_rate * duration_seconds)
    frames = bytearray()
    for frame in range(frame_count):
        value = int(8_000 * math.sin(2 * math.pi * frequency * frame / sample_rate))
        frames.extend(struct.pack("<h", value))
    output = io.BytesIO()
    with wave.open(output, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(bytes(frames))
    return output.getvalue()


def _fixture_cards() -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    ordinal = 0
    for document_id, speaker_count in _card_shape():
        for speaker_ordinal in range(1, speaker_count + 1):
            ordinal += 1
            speaker_ref = f"SPEAKER_{speaker_ordinal}"
            subject_id = None
            if ordinal == 1:
                subject_id = CHRIS_SUBJECT_ID
            elif ordinal == 2:
                subject_id = ERIC_SUBJECT_ID
            cards.append(
                {
                    "card_id": f"{document_id}::{speaker_ref}",
                    "speaker_ref": speaker_ref,
                    "proposal_label": (
                        "Synthetic enrolled-subject proposal"
                        if subject_id
                        else "Synthetic abstention"
                    ),
                    "proposal_subject_id": subject_id,
                    "confidence_band": "medium" if subject_id else "none",
                    "supporting_unit_count": 6 if subject_id else 0,
                    "opposing_unit_count": 0,
                    "transcript": (
                        f"Synthetic review fixture card {ordinal:02d}. "
                        "No person, conversation, provider, or transcript data."
                    ),
                    "audio_url": f"clips/card-{ordinal:02d}.wav",
                }
            )
    return cards


def _enrolled_options() -> list[dict[str, str]]:
    return [
        {
            "machine_identity": CHRIS_SUBJECT_ID,
            "display_label": "Enrolled subject A",
            "export_identity": CHRIS_SUBJECT_ID,
        },
        {
            "machine_identity": ERIC_SUBJECT_ID,
            "display_label": "Enrolled subject B",
            "export_identity": ERIC_SUBJECT_ID,
        },
    ]


def _render_fixture(cards: Sequence[Mapping[str, Any]]) -> str:
    return render_review_surface(
        title="Plan 0058 synthetic acoustic review",
        cards=cards,
        enrolled_options=_enrolled_options(),
        allowed_subject_ids=ALLOWED_SUBJECT_IDS,
    )


def preview_authority(
    *, repository_authority: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    authority = dict(repository_authority or _repository_authority())
    if (
        authority.get("clean") is not True
        or authority.get("upstream_ahead") != 0
        or authority.get("upstream_behind") != 0
        or not str(authority.get("commit") or "")
        or len(str(authority.get("module_sha256") or "")) != 64
    ):
        raise Plan0058Error("Repository authority must be clean and upstream-even.")
    cards = _fixture_cards()
    page = _render_fixture(cards)
    clip_hashes = {
        f"clips/card-{ordinal:02d}.wav": hashlib.sha256(
            _wav_bytes(ordinal)
        ).hexdigest()
        for ordinal in range(1, EXPECTED_CARD_COUNT + 1)
    }
    core = {
        "schema_version": PREVIEW_SCHEMA,
        "status": "ready_for_private_synthetic_fixture",
        "repository_authority": authority,
        "card_count": len(cards),
        "cards": cards,
        "allowed_subject_ids": sorted(ALLOWED_SUBJECT_IDS),
        "enrolled_options": _enrolled_options(),
        "expected_clip_sha256": clip_hashes,
        "expected_index_sha256": hashlib.sha256(page.encode()).hexdigest(),
        "negative_actions": dict(NEGATIVE_ACTIONS),
        "contains_private_audio": False,
        "contains_private_transcript": False,
        "contains_private_identity_label": False,
    }
    if core["card_count"] != EXPECTED_CARD_COUNT:
        raise Plan0058Error("The synthetic card denominator is incomplete.")
    return {**core, "content_sha256": canonical_hash(core)}


def _paths(runtime_root: Path, content_sha256: str) -> dict[str, Path]:
    root = runtime_root.expanduser().absolute()
    run = root / f"review-surface-{content_sha256[:24]}"
    return {
        "root": root,
        "run": run,
        "clips": run / "clips",
        "index": run / "index.html",
        "manifest": run / "private-manifest.json",
        "receipt": run / "receipt.json",
    }


def _write_private_bytes(path: Path, content: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(content)


def apply_fixture(
    reviewed_preview: Mapping[str, Any],
    *,
    expected_content_sha256: str,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
) -> dict[str, Any]:
    preview = dict(reviewed_preview)
    core = {key: value for key, value in preview.items() if key != "content_sha256"}
    if (
        preview.get("schema_version") != PREVIEW_SCHEMA
        or preview.get("content_sha256") != expected_content_sha256
        or canonical_hash(core) != expected_content_sha256
        or preview.get("card_count") != EXPECTED_CARD_COUNT
        or preview.get("negative_actions") != NEGATIVE_ACTIONS
        or preview.get("contains_private_audio") is not False
        or preview.get("contains_private_transcript") is not False
        or preview.get("contains_private_identity_label") is not False
        or preview.get("repository_authority") != _repository_authority()
    ):
        raise Plan0058Error("The reviewed fixture authority is stale or unsafe.")
    paths = _paths(runtime_root, expected_content_sha256)
    if paths["receipt"].exists():
        return replay_fixture(
            expected_content_sha256, runtime_root=runtime_root
        )
    if paths["run"].exists():
        raise Plan0058Error("A partial fixture run already exists.")
    ensure_private_tree(paths["root"], paths["clips"])
    for ordinal in range(1, EXPECTED_CARD_COUNT + 1):
        _write_private_bytes(
            paths["clips"] / f"card-{ordinal:02d}.wav", _wav_bytes(ordinal)
        )
    page = _render_fixture(preview["cards"])
    _write_private_bytes(paths["index"], page.encode())
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "status": "synthetic_fixture_frozen",
        "preview": preview,
    }
    write_immutable_private_json(paths["manifest"], manifest)
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "synthetic_fixture_ready_for_public_browser_proof",
        "preview_content_sha256": expected_content_sha256,
        "manifest_sha256": sha256_file(paths["manifest"]),
        "index_sha256": sha256_file(paths["index"]),
        "clip_sha256": {
            path.relative_to(paths["run"]).as_posix(): sha256_file(path)
            for path in sorted(paths["clips"].glob("*.wav"))
        },
        "card_count": EXPECTED_CARD_COUNT,
        "negative_actions": dict(NEGATIVE_ACTIONS),
        "private_directory_mode": "0700",
        "private_file_mode": "0600",
    }
    if (
        receipt["index_sha256"] != preview["expected_index_sha256"]
        or receipt["clip_sha256"] != preview["expected_clip_sha256"]
    ):
        raise Plan0058Error("The synthetic fixture bytes drifted from authority.")
    write_immutable_private_json(paths["receipt"], receipt)
    return {
        **receipt,
        "fixture_path": str(paths["run"]),
        "idempotent_replay": False,
    }


def replay_fixture(
    expected_content_sha256: str,
    *,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
) -> dict[str, Any]:
    paths = _paths(runtime_root, expected_content_sha256)
    for key in ("index", "manifest", "receipt"):
        require_private_file(paths[key], paths["root"])
    for path in paths["clips"].glob("*.wav"):
        require_private_file(path, paths["root"])
    manifest = read_private_object(paths["manifest"])
    receipt = read_private_object(paths["receipt"])
    preview = manifest.get("preview")
    if not isinstance(preview, Mapping):
        raise Plan0058Error("The frozen fixture preview is unavailable.")
    core = {key: value for key, value in preview.items() if key != "content_sha256"}
    clip_hashes = {
        path.relative_to(paths["run"]).as_posix(): sha256_file(path)
        for path in sorted(paths["clips"].glob("*.wav"))
    }
    if (
        preview.get("content_sha256") != expected_content_sha256
        or canonical_hash(core) != expected_content_sha256
        or preview.get("negative_actions") != NEGATIVE_ACTIONS
        or receipt.get("preview_content_sha256") != expected_content_sha256
        or receipt.get("manifest_sha256") != sha256_file(paths["manifest"])
        or receipt.get("index_sha256") != sha256_file(paths["index"])
        or receipt.get("index_sha256") != preview.get("expected_index_sha256")
        or receipt.get("clip_sha256") != clip_hashes
        or clip_hashes != preview.get("expected_clip_sha256")
        or len(clip_hashes) != EXPECTED_CARD_COUNT
        or paths["index"].read_text(encoding="utf-8")
        != _render_fixture(preview["cards"])
    ):
        raise Plan0058Error("The synthetic fixture authority drifted.")
    return {
        **receipt,
        "replay_schema_version": REPLAY_SCHEMA,
        "fixture_path": str(paths["run"]),
        "idempotent_replay": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the Plan 0058 synthetic review fixture.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preview")
    apply_parser = subparsers.add_parser("apply")
    apply_parser.add_argument("--expected-content-sha256", required=True)
    apply_parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    replay_parser = subparsers.add_parser("replay")
    replay_parser.add_argument("--content-sha256", required=True)
    replay_parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "preview":
        result = preview_authority()
    elif args.command == "apply":
        preview = preview_authority()
        result = apply_fixture(
            preview,
            expected_content_sha256=args.expected_content_sha256,
            runtime_root=args.runtime_root,
        )
    else:
        result = replay_fixture(
            args.content_sha256, runtime_root=args.runtime_root
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
