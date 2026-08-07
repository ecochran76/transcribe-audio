from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import acoustic_plan0057 as execution
from acoustic_audio_derivatives import (
    ensure_private_tree,
    read_private_object,
    require_private_file,
    sha256_file,
    write_immutable_private_json,
)
from acoustic_shadow_evidence import ALLOWLISTED_SUBJECT_IDS, canonical_hash


REVIEW_PREVIEW_SCHEMA = "transcribe-audio.plan0057-human-review-preview.v1"
REVIEW_MANIFEST_SCHEMA = "transcribe-audio.plan0057-human-review-manifest.v1"
REVIEW_RECEIPT_SCHEMA = "transcribe-audio.plan0057-human-review-receipt.v1"
REVIEW_REPLAY_SCHEMA = "transcribe-audio.plan0057-human-review-replay.v1"
MODULE_PATH = Path(__file__).name
DEFAULT_RUNTIME_ROOT = Path("~/.local/state/transcribe-audio/plan-0057/human-review")
EXECUTION_AUTHORITY_SHA256 = (
    "42a443a1185b31e494562a060129fae03e11e0b1a800f0863352380cd256094e"
)
EXECUTION_CONTENT_SHA256 = (
    "089d0213153bd001a86669141e3b7a0a72b7b7aa8638d71e3d8f8dc5c32b41e4"
)
EXECUTION_MANIFEST_SHA256 = (
    "29857c62105c6119095f8970de8c0dfdbba276b0267608a4fcd845e0dbecb863"
)
CHRIS_SUBJECT_ID = "subject-7c24e8f41409c6f517291fe7"
ERIC_SUBJECT_ID = "subject-df34bc192c07bd86566fff12"
NON_ENROLLED_IDENTITIES = frozenset({"neither_enrolled", "unknown"})
DISPLAY_IDENTITIES = {
    "Chris Williams": CHRIS_SUBJECT_ID,
    "Eric Cochran": ERIC_SUBJECT_ID,
    "Neither enrolled person": "neither_enrolled",
    "UNKNOWN": "unknown",
    CHRIS_SUBJECT_ID: CHRIS_SUBJECT_ID,
    ERIC_SUBJECT_ID: ERIC_SUBJECT_ID,
    "neither_enrolled": "neither_enrolled",
    "unknown": "unknown",
}
CARD_RE = re.compile(r"^([^:\s]+)::(SPEAKER_[1-9][0-9]*)$")
REVIEW_ACTION_VECTOR = {
    "record_human_review": True,
    "apply_speaker_assignments": False,
    "create_or_mutate_identities": False,
    "mutate_profiles_or_references": False,
    "write_external_provider": False,
    "enable_default_integration": False,
    "run_historical_reprocessing": False,
}


class Plan0057ReviewError(ValueError):
    """Raised when the complete human-review denominator is not trustworthy."""


def _valid_proposal(proposal: Mapping[str, Any]) -> bool:
    disposition = proposal.get("disposition")
    subject_id = proposal.get("subject_id")
    confidence = proposal.get("confidence_band")
    counts = []
    for key in (
        "supporting_unit_count",
        "supporting_candidate_family_count",
        "opposing_unit_count",
    ):
        value = proposal.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return False
        counts.append(value)
    supporting_count, family_count, opposing_count = counts
    supporting_units = proposal.get("supporting_units")
    opposing_units = proposal.get("opposing_units")
    if not isinstance(supporting_units, list) or not isinstance(opposing_units, list):
        return False

    def normalize_units(raw_units: list[Any]) -> set[tuple[str, str]] | None:
        normalized: set[tuple[str, str]] = set()
        for raw in raw_units:
            if not isinstance(raw, list) or len(raw) != 2:
                return None
            unit = (str(raw[0]), str(raw[1]))
            if unit[0] not in execution.CANDIDATE_IDS or unit[1] not in execution.METHOD_IDS:
                return None
            normalized.add(unit)
        return normalized if len(normalized) == len(raw_units) else None

    supporting = normalize_units(supporting_units)
    opposing = normalize_units(opposing_units)
    if (
        supporting is None
        or opposing is None
        or supporting & opposing
        or len(supporting) != supporting_count
        or len(opposing) != opposing_count
        or len({unit[0] for unit in supporting}) != family_count
        or supporting_count > execution.EXPECTED_THRESHOLD_UNITS
        or opposing_count > execution.EXPECTED_THRESHOLD_UNITS
        or family_count > len(execution.CANDIDATE_IDS)
        or not str(proposal.get("rationale") or "").strip()
    ):
        return False
    assign_rule = supporting_count >= 6 and family_count >= 2 and opposing_count == 0
    if disposition == "assign":
        return (
            subject_id in ALLOWLISTED_SUBJECT_IDS
            and assign_rule
            and confidence
            == ("high" if supporting_count == execution.EXPECTED_THRESHOLD_UNITS else "medium")
        )
    if disposition == "review":
        return (
            subject_id in ALLOWLISTED_SUBJECT_IDS
            and not assign_rule
            and bool(supporting_count or opposing_count)
            and confidence == "low"
        )
    return (
        disposition == "abstain"
        and subject_id is None
        and confidence == "none"
        and supporting_count == 0
        and family_count == 0
        and opposing_count == 0
    )


def _git(arguments: Sequence[str], *, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", *arguments],
        cwd=Path(__file__).resolve().parent,
        capture_output=True,
        text=not binary,
        check=False,
    )
    if result.returncode:
        raise Plan0057ReviewError("Repository authority is unavailable.")
    return result.stdout if binary else result.stdout.strip()


def _repository_authority() -> dict[str, Any]:
    if _git(["status", "--porcelain=v1", "--untracked-files=normal"]):
        raise Plan0057ReviewError("Repository must be clean.")
    if str(
        _git(["rev-list", "--left-right", "--count", "@{upstream}...HEAD"])
    ).split() != ["0", "0"]:
        raise Plan0057ReviewError("Repository must be upstream-even.")
    commit = str(_git(["rev-parse", "HEAD"]))
    body = _git(["show", f"{commit}:{MODULE_PATH}"], binary=True)
    if (
        not isinstance(body, bytes)
        or hashlib.sha256(body).hexdigest() != sha256_file(Path(__file__).resolve())
    ):
        raise Plan0057ReviewError("Committed human-review authority drifted.")
    return {
        "commit": commit,
        "module_sha256": hashlib.sha256(body).hexdigest(),
        "clean": True,
        "upstream_ahead": 0,
        "upstream_behind": 0,
    }


def _proposal_rows(execution_manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    source_results = execution_manifest.get("source_results")
    if (
        execution_manifest.get("schema_version") != execution.EXECUTION_SCHEMA
        or execution_manifest.get("status") != "complete_pending_human_review"
        or execution_manifest.get("execution_authority_content_sha256")
        != EXECUTION_AUTHORITY_SHA256
        or execution_manifest.get("content_sha256") != EXECUTION_CONTENT_SHA256
        or execution_manifest.get("eligible_recording_count") != 3
        or execution_manifest.get("entered_recording_count") != 3
        or execution_manifest.get("eligible_speaker_count") != 15
        or execution_manifest.get("covered_speaker_count") != 15
        or execution_manifest.get("identity_state_unchanged") is not True
        or execution_manifest.get("read_human_gold") is not False
        or execution_manifest.get("applied_assignments") is not False
        or execution_manifest.get("created_or_mutated_identities") is not False
        or execution_manifest.get("mutated_profiles_or_references") is not False
        or execution_manifest.get("stop_reasons") != []
        or not isinstance(source_results, list)
        or len(source_results) != 3
    ):
        raise Plan0057ReviewError("Frozen batch execution authority is invalid.")
    rows: list[dict[str, Any]] = []
    seen_cards: set[str] = set()
    for source in source_results:
        if not isinstance(source, Mapping):
            raise Plan0057ReviewError("A source result is invalid.")
        document_id = str(source.get("document_id") or "")
        conversation_key = str(source.get("conversation_key") or "")
        proposals = source.get("proposals")
        if (
            not document_id
            or not conversation_key
            or source.get("entered") is not True
            or source.get("stop_reason") is not None
            or not isinstance(proposals, list)
            or source.get("eligible_speaker_count") != len(proposals)
            or source.get("covered_speaker_count") != len(proposals)
        ):
            raise Plan0057ReviewError("A source review denominator is invalid.")
        for proposal in proposals:
            if not isinstance(proposal, Mapping):
                raise Plan0057ReviewError("An acoustic proposal is invalid.")
            speaker_ref = str(proposal.get("speaker_ref") or "")
            card_id = f"{document_id}::{speaker_ref}"
            if (
                not CARD_RE.fullmatch(card_id)
                or card_id in seen_cards
                or not _valid_proposal(proposal)
            ):
                raise Plan0057ReviewError("An acoustic proposal is unbound or unsafe.")
            seen_cards.add(card_id)
            rows.append(
                {
                    "card_id": card_id,
                    "document_id": document_id,
                    "conversation_key": conversation_key,
                    "speaker_ref": speaker_ref,
                    "proposal": dict(proposal),
                }
            )
    if len(rows) != 15:
        raise Plan0057ReviewError("Exactly 15 acoustic speakers require review.")
    return rows


def parse_review_answers(
    answer_text: str,
    *,
    expected_card_ids: Sequence[str],
) -> dict[str, dict[str, str | None]]:
    """Parse exact card-level decisions without interpreting identity clues."""

    expected = tuple(str(item) for item in expected_card_ids)
    if not expected or len(expected) != len(set(expected)):
        raise Plan0057ReviewError("Expected review card IDs are invalid.")
    answers: dict[str, dict[str, str | None]] = {}
    for raw_line in str(answer_text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "=" not in line:
            raise Plan0057ReviewError("Every review line must contain '='.")
        raw_card, raw_identity = line.split("=", 1)
        card_id = raw_card.strip()
        supplied_identity = " ".join(raw_identity.split())
        review_display_label: str | None = None
        if supplied_identity.startswith("Neither enrolled person (") and supplied_identity.endswith(")"):
            review_display_label = supplied_identity[
                len("Neither enrolled person (") : -1
            ].strip()
            if not review_display_label or len(review_display_label) > 120:
                raise Plan0057ReviewError("A non-enrolled review label is invalid.")
            actual_identity = "neither_enrolled"
        else:
            actual_identity = DISPLAY_IDENTITIES.get(supplied_identity)
        if (
            card_id not in expected
            or card_id in answers
            or actual_identity is None
        ):
            raise Plan0057ReviewError(
                "Review answers contain an unknown card, duplicate, or inexact identity."
            )
        answers[card_id] = {
            "actual_identity": actual_identity,
            "review_display_label": review_display_label,
        }
    if set(answers) != set(expected):
        raise Plan0057ReviewError("All 15 review cards require an explicit decision.")
    return answers


def preview_plan0057_review(
    answer_text: str,
    *,
    execution_manifest: Mapping[str, Any],
    repository_authority: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind complete literal decisions to the frozen batch without mutations."""

    if (
        repository_authority.get("clean") is not True
        or repository_authority.get("upstream_ahead") != 0
        or repository_authority.get("upstream_behind") != 0
    ):
        raise Plan0057ReviewError("Repository authority must be clean and upstream-even.")
    proposal_rows = _proposal_rows(execution_manifest)
    answers = parse_review_answers(
        answer_text,
        expected_card_ids=[row["card_id"] for row in proposal_rows],
    )
    decisions = []
    for row in proposal_rows:
        answer = answers[row["card_id"]]
        proposed_subject_id = row["proposal"].get("subject_id")
        actual_identity = answer["actual_identity"]
        if proposed_subject_id is None:
            proposal_decision = (
                "confirm_abstention"
                if actual_identity == "neither_enrolled"
                else "reject_abstention"
            )
        else:
            proposal_decision = (
                "confirm" if actual_identity == proposed_subject_id else "reject"
            )
        decisions.append(
            {
                "card_id": row["card_id"],
                "document_id": row["document_id"],
                "conversation_key": row["conversation_key"],
                "speaker_ref": row["speaker_ref"],
                "actual_identity": actual_identity,
                "proposed_subject_id": proposed_subject_id,
                "proposal_decision": proposal_decision,
                "review_display_label": answer["review_display_label"],
            }
        )
    core = {
        "schema_version": REVIEW_PREVIEW_SCHEMA,
        "status": "complete_human_review_ready_to_freeze",
        "execution_authority_content_sha256": EXECUTION_AUTHORITY_SHA256,
        "execution_content_sha256": EXECUTION_CONTENT_SHA256,
        "execution_manifest_sha256": EXECUTION_MANIFEST_SHA256,
        "repository_authority": dict(repository_authority),
        "recording_count": 3,
        "speaker_count": 15,
        "decision_count": len(decisions),
        "decisions": decisions,
        "review_complete": True,
        "contains_display_names": any(
            item["review_display_label"] is not None for item in decisions
        ),
        "display_names_are_review_attributes_only": True,
        "action_vector": dict(REVIEW_ACTION_VECTOR),
    }
    return {**core, "content_sha256": canonical_hash(core)}


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
        "execution_content_sha256": preview["execution_content_sha256"],
        "decision_count": preview["decision_count"],
        "review_complete": True,
        "applied_assignments": False,
        "created_or_mutated_identities": False,
        "mutated_profiles_or_references": False,
        "wrote_external_provider": False,
        "mode": "0600",
    }


def freeze_plan0057_review(
    reviewed_preview: Mapping[str, Any],
    *,
    expected_content_sha256: str,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
) -> dict[str, Any]:
    preview = dict(reviewed_preview)
    core = {key: value for key, value in preview.items() if key != "content_sha256"}
    if (
        preview.get("schema_version") != REVIEW_PREVIEW_SCHEMA
        or preview.get("content_sha256") != expected_content_sha256
        or canonical_hash(core) != expected_content_sha256
        or preview.get("review_complete") is not True
        or preview.get("decision_count") != 15
        or preview.get("execution_authority_content_sha256")
        != EXECUTION_AUTHORITY_SHA256
        or preview.get("execution_content_sha256") != EXECUTION_CONTENT_SHA256
        or preview.get("execution_manifest_sha256") != EXECUTION_MANIFEST_SHA256
        or preview.get("action_vector") != REVIEW_ACTION_VECTOR
    ):
        raise Plan0057ReviewError("Reviewed human decisions are stale or incomplete.")
    paths = _review_paths(runtime_root, expected_content_sha256)
    if paths["receipt"].exists():
        return replay_plan0057_review(
            expected_content_sha256,
            runtime_root=runtime_root,
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


def replay_plan0057_review(
    expected_content_sha256: str,
    *,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
) -> dict[str, Any]:
    paths = _review_paths(runtime_root, expected_content_sha256)
    require_private_file(paths["manifest"], paths["root"])
    require_private_file(paths["receipt"], paths["root"])
    manifest = read_private_object(paths["manifest"])
    receipt = read_private_object(paths["receipt"])
    preview = manifest.get("preview")
    if not isinstance(preview, Mapping):
        raise Plan0057ReviewError("Frozen human review is invalid.")
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
        or canonical_hash(core) != expected_content_sha256
        or preview.get("action_vector") != REVIEW_ACTION_VECTOR
    ):
        raise Plan0057ReviewError("Frozen human review evidence drifted.")
    return {
        **receipt,
        "replay_schema_version": REVIEW_REPLAY_SCHEMA,
        "idempotent_replay": True,
    }


def _live_execution_manifest() -> dict[str, Any]:
    replay = execution.replay_execution(EXECUTION_AUTHORITY_SHA256)
    paths = execution._execution_paths(
        execution.DEFAULT_RUNTIME_ROOT,
        EXECUTION_AUTHORITY_SHA256,
    )
    require_private_file(paths["manifest"], paths["root"])
    manifest = read_private_object(paths["manifest"])
    if (
        replay.get("idempotent_replay") is not True
        or sha256_file(paths["manifest"]) != EXECUTION_MANIFEST_SHA256
        or manifest.get("content_sha256") != EXECUTION_CONTENT_SHA256
    ):
        raise Plan0057ReviewError("Frozen batch execution evidence drifted.")
    return manifest


def build_live_review_preview(answer_text: str) -> dict[str, Any]:
    return preview_plan0057_review(
        answer_text,
        execution_manifest=_live_execution_manifest(),
        repository_authority=_repository_authority(),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Freeze or replay Plan 0057 human review.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("preview", "freeze"):
        child = subparsers.add_parser(command)
        child.add_argument("--answers-file", type=Path, required=True)
        child.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
        if command == "freeze":
            child.add_argument("--expected-content-sha256", required=True)
    replay = subparsers.add_parser("replay")
    replay.add_argument("--review-content-sha256", required=True)
    replay.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "replay":
        result = replay_plan0057_review(
            args.review_content_sha256,
            runtime_root=args.runtime_root,
        )
    else:
        answer_text = args.answers_file.read_text(encoding="utf-8")
        preview = build_live_review_preview(answer_text)
        if args.command == "preview":
            result = preview
        else:
            result = freeze_plan0057_review(
                preview,
                expected_content_sha256=args.expected_content_sha256,
                runtime_root=args.runtime_root,
            )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
