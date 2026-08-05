"""Independent metric recomputation for the Plan 0056 acoustic pilot."""

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
import acoustic_plan0056_review as review
import acoustic_plan0056_runner as runner


AUDIT_SCHEMA = "transcribe-audio.plan0056-independent-audit.v1"
TERMINAL_PREVIEW_SCHEMA = "transcribe-audio.plan0056-terminal-audit-preview.v1"
TERMINAL_MANIFEST_SCHEMA = "transcribe-audio.plan0056-terminal-audit-manifest.v1"
TERMINAL_RECEIPT_SCHEMA = "transcribe-audio.plan0056-terminal-audit-receipt.v1"
TERMINAL_REPLAY_SCHEMA = "transcribe-audio.plan0056-terminal-audit-replay.v1"
MODULE_PATH = Path(__file__).name
DEFAULT_RUNTIME_ROOT = Path("~/.local/state/transcribe-audio/plan-0056/terminal-audit")
SHA256_RE = re.compile(r"[a-f0-9]{64}")
ALLOWLIST = {
    "subject-7c24e8f41409c6f517291fe7",
    "subject-df34bc192c07bd86566fff12",
}
NON_ENROLLED_IDENTITIES = {"neither_enrolled", "unknown"}


class Plan0056AuditError(ValueError):
    """Raised when independent pilot recomputation finds incomplete evidence."""


def _git(arguments: Sequence[str], *, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", *arguments], cwd=Path(__file__).resolve().parent,
        capture_output=True, text=not binary, check=False,
    )
    if result.returncode:
        raise Plan0056AuditError("Repository authority is unavailable.")
    return result.stdout if binary else result.stdout.strip()


def _repository_authority() -> dict[str, Any]:
    if _git(["status", "--porcelain=v1", "--untracked-files=normal"]):
        raise Plan0056AuditError("Repository must be clean.")
    if str(_git(["rev-list", "--left-right", "--count", "@{upstream}...HEAD"])).split() != ["0", "0"]:
        raise Plan0056AuditError("Repository must be upstream-even.")
    commit = str(_git(["rev-parse", "HEAD"]))
    body = _git(["show", f"{commit}:{MODULE_PATH}"], binary=True)
    if not isinstance(body, bytes) or hashlib.sha256(body).hexdigest() != sha256_file(Path(__file__).resolve()):
        raise Plan0056AuditError("Committed independent-audit authority drifted.")
    return {
        "commit": commit, "module_sha256": hashlib.sha256(body).hexdigest(),
        "clean": True, "upstream_ahead": 0, "upstream_behind": 0,
    }


def _canonical_hash(value: Any) -> str:
    body = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def recompute_plan0056_audit(
    *,
    execution_manifest: Mapping[str, Any],
    review_preview: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute every pilot denominator without trusting runner summaries."""

    proposals_evidence = execution_manifest.get("artifacts", {}).get("proposals", {})
    proposals = proposals_evidence.get("proposals")
    decisions = review_preview.get("decisions")
    if (
        execution_manifest.get("identity_state_unchanged") is not True
        or execution_manifest.get("read_pilot_outcome_gold") is not False
        or execution_manifest.get("applied_assignments") is not False
        or set(proposals_evidence.get("allowlisted_subject_ids") or []) != ALLOWLIST
        or review_preview.get("review_complete") is not True
        or review_preview.get("decision_count") != 2
        or not isinstance(proposals, list)
        or not isinstance(decisions, list)
        or len(proposals) != 2
        or len(decisions) != 2
    ):
        raise Plan0056AuditError("Pilot guard or review denominator is incomplete.")
    proposals_by_ref = {
        str(item.get("speaker_ref") or ""): item
        for item in proposals
        if isinstance(item, Mapping)
    }
    decisions_by_ref = {
        str(item.get("speaker_ref") or ""): item
        for item in decisions
        if isinstance(item, Mapping)
    }
    if (
        len(proposals_by_ref) != 2
        or proposals_by_ref.keys() != decisions_by_ref.keys()
    ):
        raise Plan0056AuditError("Proposal and human-review references differ.")

    rows = []
    for speaker_ref in sorted(proposals_by_ref):
        proposal = proposals_by_ref[speaker_ref]
        decision = decisions_by_ref[speaker_ref]
        disposition = proposal.get("disposition")
        proposed_subject = proposal.get("subject_id")
        actual_identity = decision.get("actual_identity")
        if (
            disposition not in {"assign", "review", "abstain"}
            or (disposition != "abstain" and proposed_subject not in ALLOWLIST)
            or (disposition == "abstain" and proposed_subject is not None)
            or actual_identity not in ALLOWLIST | NON_ENROLLED_IDENTITIES
            or decision.get("proposed_subject_id") != proposed_subject
        ):
            raise Plan0056AuditError("A proposal or review identity is invalid.")
        expected_review = (
            "confirm" if proposed_subject is not None and proposed_subject == actual_identity
            else "reject"
        )
        if decision.get("proposal_decision") != expected_review:
            raise Plan0056AuditError("A human review decision is internally inconsistent.")
        confirmed = expected_review == "confirm"
        assignment = disposition == "assign"
        correct_assignment = assignment and confirmed
        wrong_assignment = assignment and not confirmed
        rows.append(
            {
                "speaker_ref": speaker_ref,
                "disposition": disposition,
                "confidence_band": proposal.get("confidence_band"),
                "proposed_subject_id": proposed_subject,
                "actual_identity": actual_identity,
                "proposal_confirmed": confirmed,
                "correct_assignment": correct_assignment,
                "wrong_assignment": wrong_assignment,
                "high_confidence_wrong": (
                    wrong_assignment and proposal.get("confidence_band") == "high"
                ),
                "is_enrolled": actual_identity in ALLOWLIST,
            }
        )

    enrolled = [row for row in rows if row["is_enrolled"]]
    proposal_count = sum(row["proposed_subject_id"] is not None for row in rows)
    confirmed_count = sum(row["proposal_confirmed"] for row in rows)
    enrolled_correct = sum(row["correct_assignment"] for row in enrolled)
    metrics = {
        "speaker_count": len(rows),
        "enrolled_speaker_count": len(enrolled),
        "proposal_count": proposal_count,
        "proposal_confirmed_count": confirmed_count,
        "proposal_rejected_count": proposal_count - confirmed_count,
        "assign_disposition_count": sum(row["disposition"] == "assign" for row in rows),
        "correct_assignment_count": sum(row["correct_assignment"] for row in rows),
        "wrong_assignment_count": sum(row["wrong_assignment"] for row in rows),
        "high_confidence_wrong_count": sum(row["high_confidence_wrong"] for row in rows),
        "review_count": sum(row["disposition"] == "review" for row in rows),
        "abstention_count": sum(row["disposition"] == "abstain" for row in rows),
        "enrolled_correct_assignment_count": enrolled_correct,
        "enrolled_recall": enrolled_correct / len(enrolled) if enrolled else 0.0,
        "proposal_precision": confirmed_count / proposal_count if proposal_count else 0.0,
        "identity_creation_count": 0,
        "profile_or_reference_mutation_count": 0,
    }
    if metrics["high_confidence_wrong_count"]:
        terminal_decision = "stop"
    elif metrics["wrong_assignment_count"] or metrics["enrolled_recall"] < 1.0:
        terminal_decision = "refine"
    else:
        terminal_decision = "plan_next_bounded_integration_milestone"
    core = {
        "schema_version": AUDIT_SCHEMA,
        "status": "independent_recomputation_complete",
        "rows": rows,
        "metrics": metrics,
        "terminal_decision": terminal_decision,
        "independent_guard_recomputed": True,
        "action_vector": {
            "freeze_terminal_decision": True,
            "apply_speaker_assignments": False,
            "create_or_mutate_identities": False,
            "mutate_profiles_or_references": False,
            "enable_default_integration": False,
            "run_historical_reprocessing": False,
        },
    }
    return {**core, "content_sha256": _canonical_hash(core)}


def preview_plan0056_terminal_audit(
    *,
    execution_manifest: Mapping[str, Any],
    review_preview: Mapping[str, Any],
    repository_authority: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        repository_authority.get("clean") is not True
        or repository_authority.get("upstream_ahead") != 0
        or repository_authority.get("upstream_behind") != 0
    ):
        raise Plan0056AuditError("Repository authority must be clean and upstream-even.")
    audit = recompute_plan0056_audit(
        execution_manifest=execution_manifest, review_preview=review_preview
    )
    core = {
        "schema_version": TERMINAL_PREVIEW_SCHEMA,
        "status": "independent_terminal_audit_ready_to_freeze",
        "repository_authority": dict(repository_authority),
        "audit": audit,
        "audit_content_sha256": audit["content_sha256"],
        "metrics": audit["metrics"],
        "terminal_decision": audit["terminal_decision"],
        "action_vector": audit["action_vector"],
        "applied_assignments": False,
    }
    return {**core, "content_sha256": _canonical_hash(core)}


def _terminal_paths(runtime_root: Path, content_sha256: str) -> dict[str, Path]:
    root = runtime_root.expanduser().absolute()
    run = root / f"terminal-audit-{content_sha256[:24]}"
    return {
        "root": root,
        "run": run,
        "manifest": run / "private-manifest.json",
        "receipt": run / "receipt.json",
    }


def _terminal_receipt(preview: Mapping[str, Any], manifest_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": TERMINAL_RECEIPT_SCHEMA,
        "status": "terminal_decision_frozen",
        "preview_content_sha256": preview["content_sha256"],
        "manifest_sha256": manifest_sha256,
        "audit_content_sha256": preview["audit_content_sha256"],
        "terminal_decision": preview["terminal_decision"],
        "speaker_count": preview["metrics"]["speaker_count"],
        "enrolled_speaker_count": preview["metrics"]["enrolled_speaker_count"],
        "correct_assignment_count": preview["metrics"]["correct_assignment_count"],
        "wrong_assignment_count": preview["metrics"]["wrong_assignment_count"],
        "high_confidence_wrong_count": preview["metrics"][
            "high_confidence_wrong_count"
        ],
        "applied_assignments": False,
        "mode": "0600",
    }


def freeze_plan0056_terminal_audit(
    reviewed_preview: Mapping[str, Any],
    *,
    expected_content_sha256: str,
    runtime_root: Path,
) -> dict[str, Any]:
    preview = dict(reviewed_preview)
    core = {key: value for key, value in preview.items() if key != "content_sha256"}
    if (
        preview.get("schema_version") != TERMINAL_PREVIEW_SCHEMA
        or preview.get("content_sha256") != expected_content_sha256
        or _canonical_hash(core) != expected_content_sha256
        or preview.get("applied_assignments") is not False
    ):
        raise Plan0056AuditError("Reviewed terminal audit is stale or unsafe.")
    paths = _terminal_paths(runtime_root, expected_content_sha256)
    if paths["receipt"].exists():
        return replay_plan0056_terminal_audit(
            expected_content_sha256, runtime_root=runtime_root
        )
    ensure_private_tree(paths["root"], paths["run"])
    manifest = {
        "schema_version": TERMINAL_MANIFEST_SCHEMA,
        "status": "terminal_decision_frozen",
        "preview": preview,
    }
    write_immutable_private_json(paths["manifest"], manifest)
    receipt = _terminal_receipt(preview, sha256_file(paths["manifest"]))
    write_immutable_private_json(paths["receipt"], receipt)
    return {**receipt, "idempotent_replay": False}


def replay_plan0056_terminal_audit(
    expected_content_sha256: str,
    *,
    runtime_root: Path,
) -> dict[str, Any]:
    paths = _terminal_paths(runtime_root, expected_content_sha256)
    require_private_file(paths["manifest"], paths["root"])
    require_private_file(paths["receipt"], paths["root"])
    manifest = read_private_object(paths["manifest"])
    receipt = read_private_object(paths["receipt"])
    preview = manifest.get("preview")
    if not isinstance(preview, Mapping):
        raise Plan0056AuditError("Frozen terminal audit is invalid.")
    core = {key: value for key, value in preview.items() if key != "content_sha256"}
    expected_manifest = {
        "schema_version": TERMINAL_MANIFEST_SCHEMA,
        "status": "terminal_decision_frozen",
        "preview": dict(preview),
    }
    expected_receipt = _terminal_receipt(preview, sha256_file(paths["manifest"]))
    if (
        manifest != expected_manifest
        or receipt != expected_receipt
        or preview.get("content_sha256") != expected_content_sha256
        or _canonical_hash(core) != expected_content_sha256
        or preview.get("audit", {}).get("content_sha256")
        != preview.get("audit_content_sha256")
    ):
        raise Plan0056AuditError("Frozen terminal audit evidence drifted.")
    return {
        **receipt,
        "replay_schema_version": TERMINAL_REPLAY_SCHEMA,
        "idempotent_replay": True,
    }


def build_live_terminal_audit_preview(review_content_sha256: str) -> dict[str, Any]:
    review.replay_plan0056_review(
        review_content_sha256, runtime_root=review.DEFAULT_RUNTIME_ROOT
    )
    review_paths = review._review_paths(review.DEFAULT_RUNTIME_ROOT, review_content_sha256)
    review_manifest = read_private_object(review_paths["manifest"])
    review_preview = review_manifest.get("preview")
    execution_paths = runner._execution_paths(
        runner.DEFAULT_RUNTIME_ROOT, review.EXECUTION_AUTHORITY_SHA256
    )
    runner.replay_local_pilot(review.EXECUTION_AUTHORITY_SHA256)
    execution_manifest = read_private_object(execution_paths["manifest"])
    if not isinstance(review_preview, Mapping) or not isinstance(execution_manifest, Mapping):
        raise Plan0056AuditError("Frozen review or execution evidence is invalid.")
    return preview_plan0056_terminal_audit(
        execution_manifest=execution_manifest,
        review_preview=review_preview,
        repository_authority=_repository_authority(),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recompute and freeze Plan 0056 terminal audit.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("preview", "freeze"):
        child = subparsers.add_parser(command)
        child.add_argument("--review-content-sha256", required=True)
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
            raise Plan0056AuditError("Terminal-audit content hash is invalid.")
        result = replay_plan0056_terminal_audit(
            args.content_sha256, runtime_root=args.runtime_root
        )
    else:
        if not SHA256_RE.fullmatch(args.review_content_sha256):
            raise Plan0056AuditError("Human-review content hash is invalid.")
        preview = build_live_terminal_audit_preview(args.review_content_sha256)
        if args.command == "preview":
            result = preview
        else:
            if args.expected_content_sha256 != preview["content_sha256"]:
                raise Plan0056AuditError("Reviewed terminal-audit hash is stale.")
            result = freeze_plan0056_terminal_audit(
                preview, expected_content_sha256=args.expected_content_sha256,
                runtime_root=args.runtime_root,
            )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
