"""Immutable human-review capture for the enrolled-only Plan 0056 pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
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
import acoustic_plan0056_runner as runner


REVIEW_PREVIEW_SCHEMA = "transcribe-audio.plan0056-human-review-preview.v1"
REVIEW_MANIFEST_SCHEMA = "transcribe-audio.plan0056-human-review-manifest.v1"
REVIEW_RECEIPT_SCHEMA = "transcribe-audio.plan0056-human-review-receipt.v1"
REVIEW_REPLAY_SCHEMA = "transcribe-audio.plan0056-human-review-replay.v1"
MODULE_PATH = Path(__file__).name
DEFAULT_RUNTIME_ROOT = Path("~/.local/state/transcribe-audio/plan-0056/human-review")
EXECUTION_AUTHORITY_SHA256 = (
    "67e667eae5440738e4cea05e457d2ddce386dcbefb74d8f8ade9ca2c8b84a8ca"
)
PROPOSAL_CONTENT_SHA256 = (
    "8268c506906267883334af3f8fedf94369bb6fb94de1e33dd11a16ee0debb16f"
)
EXECUTION_MANIFEST_SHA256 = (
    "c54564c19f8f06949ec4300f0d5fa637c6b86b42102df29669a8d3377174ab73"
)
SHA256_RE = re.compile(r"[a-f0-9]{64}")
CHRIS_SUBJECT_ID = "subject-7c24e8f41409c6f517291fe7"
ERIC_SUBJECT_ID = "subject-df34bc192c07bd86566fff12"
_DISPLAY_REVIEW_IDENTITIES = {
    "Chris Williams": CHRIS_SUBJECT_ID,
    "Eric Cochran": ERIC_SUBJECT_ID,
    "Neither enrolled person": "neither_enrolled",
    "UNKNOWN": "unknown",
}


class Plan0056ReviewError(ValueError):
    """Raised when the complete human-review denominator is not trustworthy."""


def _git(arguments: Sequence[str], *, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", *arguments], cwd=Path(__file__).resolve().parent,
        capture_output=True, text=not binary, check=False,
    )
    if result.returncode:
        raise Plan0056ReviewError("Repository authority is unavailable.")
    return result.stdout if binary else result.stdout.strip()


def _repository_authority() -> dict[str, Any]:
    if _git(["status", "--porcelain=v1", "--untracked-files=normal"]):
        raise Plan0056ReviewError("Repository must be clean.")
    if str(_git(["rev-list", "--left-right", "--count", "@{upstream}...HEAD"])).split() != ["0", "0"]:
        raise Plan0056ReviewError("Repository must be upstream-even.")
    commit = str(_git(["rev-parse", "HEAD"]))
    body = _git(["show", f"{commit}:{MODULE_PATH}"], binary=True)
    if not isinstance(body, bytes) or hashlib.sha256(body).hexdigest() != sha256_file(Path(__file__).resolve()):
        raise Plan0056ReviewError("Committed human-review authority drifted.")
    return {
        "commit": commit, "module_sha256": hashlib.sha256(body).hexdigest(),
        "clean": True, "upstream_ahead": 0, "upstream_behind": 0,
    }


def _canonical_hash(value: Any) -> str:
    body = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _parse_answers(answer_text: str) -> dict[str, dict[str, str | None]]:
    answers: dict[str, dict[str, str | None]] = {}
    for raw_line in answer_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "=" not in line:
            raise Plan0056ReviewError("Every review line must contain '='.")
        raw_ref, raw_identity = line.split("=", 1)
        display_ref = raw_ref.strip()
        speaker_ref = {"Speaker 1": "SPEAKER_1", "Speaker 2": "SPEAKER_2"}.get(
            display_ref
        )
        supplied_identity = raw_identity.strip()
        review_display_label: str | None = None
        if supplied_identity.startswith("Neither enrolled person (") and supplied_identity.endswith(")"):
            review_display_label = supplied_identity[len("Neither enrolled person ("):-1].strip()
            if not review_display_label or len(review_display_label) > 120:
                raise Plan0056ReviewError("A non-enrolled review label is invalid.")
            identity = "neither_enrolled"
        else:
            identity = _DISPLAY_REVIEW_IDENTITIES.get(supplied_identity)
        if not speaker_ref or not identity or speaker_ref in answers:
            raise Plan0056ReviewError(
                "Review answers are incomplete, duplicated, or not exact identities."
            )
        answers[speaker_ref] = {
            "actual_identity": identity,
            "review_display_label": review_display_label,
        }
    if set(answers) != {"SPEAKER_1", "SPEAKER_2"}:
        raise Plan0056ReviewError("Both pilot speakers require an explicit decision.")
    return answers


def preview_plan0056_review(
    answer_text: str,
    *,
    execution_manifest: Mapping[str, Any],
    repository_authority: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve exact operator answers into stable IDs without mutating identity state."""

    if (
        execution_manifest.get("authority_content_sha256")
        != EXECUTION_AUTHORITY_SHA256
        or execution_manifest.get("identity_state_unchanged") is not True
        or execution_manifest.get("read_pilot_outcome_gold") is not False
        or execution_manifest.get("applied_assignments") is not False
    ):
        raise Plan0056ReviewError("Frozen pilot execution authority is invalid.")
    proposals = execution_manifest.get("artifacts", {}).get("proposals", {})
    if (
        proposals.get("content_sha256") != PROPOSAL_CONTENT_SHA256
        or set(proposals.get("allowlisted_subject_ids") or [])
        != {CHRIS_SUBJECT_ID, ERIC_SUBJECT_ID}
    ):
        raise Plan0056ReviewError("Frozen pilot proposals drifted.")
    if (
        repository_authority.get("clean") is not True
        or repository_authority.get("upstream_ahead") != 0
        or repository_authority.get("upstream_behind") != 0
    ):
        raise Plan0056ReviewError("Repository authority must be clean and upstream-even.")
    answers = _parse_answers(answer_text)
    raw_proposals = proposals.get("proposals")
    if not isinstance(raw_proposals, list) or len(raw_proposals) != 2:
        raise Plan0056ReviewError("Every pilot proposal must receive review.")
    by_ref = {str(item.get("speaker_ref") or ""): item for item in raw_proposals}
    if set(by_ref) != set(answers):
        raise Plan0056ReviewError("Pilot proposal and review denominators differ.")
    decisions = []
    for speaker_ref in ("SPEAKER_1", "SPEAKER_2"):
        proposal = by_ref[speaker_ref]
        proposed_subject_id = proposal.get("subject_id")
        if proposed_subject_id not in {CHRIS_SUBJECT_ID, ERIC_SUBJECT_ID}:
            raise Plan0056ReviewError("A non-abstaining proposal is not allowlisted.")
        actual_identity = answers[speaker_ref]["actual_identity"]
        decisions.append(
            {
                "speaker_ref": speaker_ref,
                "actual_identity": actual_identity,
                "proposal_decision": (
                    "confirm" if actual_identity == proposed_subject_id else "reject"
                ),
                "proposed_subject_id": proposed_subject_id,
                "review_display_label": answers[speaker_ref]["review_display_label"],
            }
        )
    core = {
        "schema_version": REVIEW_PREVIEW_SCHEMA,
        "status": "complete_human_review_ready_to_freeze",
        "execution_authority_sha256": EXECUTION_AUTHORITY_SHA256,
        "proposal_content_sha256": PROPOSAL_CONTENT_SHA256,
        "repository_authority": dict(repository_authority),
        "speaker_count": 2,
        "decision_count": len(decisions),
        "decisions": decisions,
        "review_complete": True,
        "contains_display_names": any(
            item["review_display_label"] is not None for item in decisions
        ),
        "display_names_are_review_attributes_only": True,
        "action_vector": {
            "record_human_review": True,
            "apply_speaker_assignments": False,
            "create_or_mutate_identities": False,
            "mutate_profiles_or_references": False,
            "write_external_provider": False,
        },
    }
    return {**core, "content_sha256": _canonical_hash(core)}


def _review_paths(runtime_root: Path, content_sha256: str) -> dict[str, Path]:
    root = runtime_root.expanduser().absolute()
    run = root / f"human-review-{content_sha256[:24]}"
    return {
        "root": root,
        "run": run,
        "manifest": run / "private-manifest.json",
        "receipt": run / "receipt.json",
    }


def _review_receipt(preview: Mapping[str, Any], manifest_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": REVIEW_RECEIPT_SCHEMA,
        "status": "human_review_frozen",
        "preview_content_sha256": preview["content_sha256"],
        "manifest_sha256": manifest_sha256,
        "execution_authority_sha256": preview["execution_authority_sha256"],
        "proposal_content_sha256": preview["proposal_content_sha256"],
        "decision_count": preview["decision_count"],
        "review_complete": True,
        "applied_assignments": False,
        "created_or_mutated_identities": False,
        "mutated_profiles_or_references": False,
        "wrote_external_provider": False,
        "mode": "0600",
    }


def freeze_plan0056_review(
    reviewed_preview: Mapping[str, Any],
    *,
    expected_content_sha256: str,
    runtime_root: Path,
) -> dict[str, Any]:
    preview = dict(reviewed_preview)
    core = {key: value for key, value in preview.items() if key != "content_sha256"}
    if (
        preview.get("schema_version") != REVIEW_PREVIEW_SCHEMA
        or preview.get("content_sha256") != expected_content_sha256
        or _canonical_hash(core) != expected_content_sha256
        or preview.get("review_complete") is not True
        or preview.get("decision_count") != 2
    ):
        raise Plan0056ReviewError("Reviewed human decisions are stale or incomplete.")
    paths = _review_paths(runtime_root, expected_content_sha256)
    if paths["receipt"].exists():
        return replay_plan0056_review(
            expected_content_sha256, runtime_root=runtime_root
        )
    ensure_private_tree(paths["root"], paths["run"])
    manifest = {
        "schema_version": REVIEW_MANIFEST_SCHEMA,
        "status": "human_review_frozen",
        "preview": preview,
    }
    write_immutable_private_json(paths["manifest"], manifest)
    receipt = _review_receipt(preview, sha256_file(paths["manifest"]))
    write_immutable_private_json(paths["receipt"], receipt)
    return {**receipt, "idempotent_replay": False}


def replay_plan0056_review(
    expected_content_sha256: str,
    *,
    runtime_root: Path,
) -> dict[str, Any]:
    paths = _review_paths(runtime_root, expected_content_sha256)
    require_private_file(paths["manifest"], paths["root"])
    require_private_file(paths["receipt"], paths["root"])
    manifest = read_private_object(paths["manifest"])
    receipt = read_private_object(paths["receipt"])
    preview = manifest.get("preview")
    if not isinstance(preview, Mapping):
        raise Plan0056ReviewError("Frozen human review is invalid.")
    core = {key: value for key, value in preview.items() if key != "content_sha256"}
    expected_manifest = {
        "schema_version": REVIEW_MANIFEST_SCHEMA,
        "status": "human_review_frozen",
        "preview": dict(preview),
    }
    expected_receipt = _review_receipt(preview, sha256_file(paths["manifest"]))
    if (
        manifest != expected_manifest
        or receipt != expected_receipt
        or preview.get("content_sha256") != expected_content_sha256
        or _canonical_hash(core) != expected_content_sha256
    ):
        raise Plan0056ReviewError("Frozen human review evidence drifted.")
    return {
        **receipt,
        "replay_schema_version": REVIEW_REPLAY_SCHEMA,
        "idempotent_replay": True,
    }


def _live_execution_manifest() -> dict[str, Any]:
    replay = runner.replay_local_pilot(EXECUTION_AUTHORITY_SHA256)
    paths = runner._execution_paths(runner.DEFAULT_RUNTIME_ROOT, EXECUTION_AUTHORITY_SHA256)
    if replay.get("idempotent_replay") is not True or sha256_file(paths["manifest"]) != EXECUTION_MANIFEST_SHA256:
        raise Plan0056ReviewError("Frozen pilot execution evidence drifted.")
    value = read_private_object(paths["manifest"])
    if not isinstance(value, dict):
        raise Plan0056ReviewError("Frozen pilot execution evidence is invalid.")
    return value


def build_live_review_preview(answer_text: str) -> dict[str, Any]:
    return preview_plan0056_review(
        answer_text,
        execution_manifest=_live_execution_manifest(),
        repository_authority=_repository_authority(),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Freeze and replay Plan 0056 human review.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("preview", "freeze"):
        child = subparsers.add_parser(command)
        child.add_argument("--answers-file", required=True, type=Path)
        if command == "freeze":
            child.add_argument("--expected-content-sha256", required=True)
            child.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    replay = subparsers.add_parser("replay")
    replay.add_argument("--content-sha256", required=True)
    replay.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "replay":
        if not SHA256_RE.fullmatch(args.content_sha256):
            raise Plan0056ReviewError("Human-review content hash is invalid.")
        result = replay_plan0056_review(args.content_sha256, runtime_root=args.runtime_root)
    else:
        answer_text = args.answers_file.read_text(encoding="utf-8")
        preview = build_live_review_preview(answer_text)
        if args.command == "preview":
            result = preview
        else:
            if args.expected_content_sha256 != preview["content_sha256"]:
                raise Plan0056ReviewError("Reviewed human-review hash is stale.")
            result = freeze_plan0056_review(
                preview, expected_content_sha256=args.expected_content_sha256,
                runtime_root=args.runtime_root,
            )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
