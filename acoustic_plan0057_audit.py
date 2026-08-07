from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import acoustic_plan0057 as execution
import acoustic_plan0057_review as review
from acoustic_audio_derivatives import (
    ensure_private_tree,
    read_private_object,
    require_private_file,
    sha256_file,
    write_immutable_private_json,
)
from acoustic_shadow_evidence import ALLOWLISTED_SUBJECT_IDS, canonical_hash


AUDIT_SCHEMA = "transcribe-audio.plan0057-independent-audit.v1"
TERMINAL_PREVIEW_SCHEMA = "transcribe-audio.plan0057-terminal-audit-preview.v1"
TERMINAL_MANIFEST_SCHEMA = "transcribe-audio.plan0057-terminal-audit-manifest.v1"
TERMINAL_RECEIPT_SCHEMA = "transcribe-audio.plan0057-terminal-audit-receipt.v1"
TERMINAL_REPLAY_SCHEMA = "transcribe-audio.plan0057-terminal-audit-replay.v1"
MODULE_PATH = Path(__file__).name
DEFAULT_RUNTIME_ROOT = Path("~/.local/state/transcribe-audio/plan-0057/terminal-audit")
NON_ENROLLED_IDENTITIES = frozenset({"neither_enrolled", "unknown"})
AUDIT_ACTION_VECTOR = {
    "freeze_terminal_decision": True,
    "apply_speaker_assignments": False,
    "create_or_mutate_identities": False,
    "mutate_profiles_or_references": False,
    "write_external_provider": False,
    "enable_default_integration": False,
    "run_historical_reprocessing": False,
}


class Plan0057AuditError(ValueError):
    """Raised when independent recomputation finds incomplete evidence."""


def _valid_audit_proposal(proposal: Mapping[str, Any]) -> bool:
    values = []
    for key in (
        "supporting_unit_count",
        "supporting_candidate_family_count",
        "opposing_unit_count",
    ):
        value = proposal.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return False
        values.append(value)
    supporting_count, family_count, opposing_count = values
    supporting_units = proposal.get("supporting_units")
    opposing_units = proposal.get("opposing_units")
    if not isinstance(supporting_units, list) or not isinstance(opposing_units, list):
        return False

    def units(raw_values: list[Any]) -> set[tuple[str, str]] | None:
        result: set[tuple[str, str]] = set()
        for raw in raw_values:
            if not isinstance(raw, list) or len(raw) != 2:
                return None
            value = (str(raw[0]), str(raw[1]))
            if value[0] not in execution.CANDIDATE_IDS or value[1] not in execution.METHOD_IDS:
                return None
            result.add(value)
        return result if len(result) == len(raw_values) else None

    supporting = units(supporting_units)
    opposing = units(opposing_units)
    if (
        supporting is None
        or opposing is None
        or supporting & opposing
        or len(supporting) != supporting_count
        or len(opposing) != opposing_count
        or len({value[0] for value in supporting}) != family_count
        or not str(proposal.get("rationale") or "").strip()
    ):
        return False
    disposition = proposal.get("disposition")
    subject_id = proposal.get("subject_id")
    confidence = proposal.get("confidence_band")
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
        raise Plan0057AuditError("Repository authority is unavailable.")
    return result.stdout if binary else result.stdout.strip()


def _repository_authority() -> dict[str, Any]:
    if _git(["status", "--porcelain=v1", "--untracked-files=normal"]):
        raise Plan0057AuditError("Repository must be clean.")
    if str(
        _git(["rev-list", "--left-right", "--count", "@{upstream}...HEAD"])
    ).split() != ["0", "0"]:
        raise Plan0057AuditError("Repository must be upstream-even.")
    commit = str(_git(["rev-parse", "HEAD"]))
    body = _git(["show", f"{commit}:{MODULE_PATH}"], binary=True)
    if (
        not isinstance(body, bytes)
        or hashlib.sha256(body).hexdigest() != sha256_file(Path(__file__).resolve())
    ):
        raise Plan0057AuditError("Committed independent-audit authority drifted.")
    return {
        "commit": commit,
        "module_sha256": hashlib.sha256(body).hexdigest(),
        "clean": True,
        "upstream_ahead": 0,
        "upstream_behind": 0,
    }


def _indexed_execution_rows(
    execution_manifest: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    if (
        execution_manifest.get("schema_version") != execution.EXECUTION_SCHEMA
        or execution_manifest.get("status") != "complete_pending_human_review"
        or execution_manifest.get("execution_authority_content_sha256")
        != review.EXECUTION_AUTHORITY_SHA256
        or execution_manifest.get("content_sha256")
        != review.EXECUTION_CONTENT_SHA256
        or execution_manifest.get("identity_state_unchanged") is not True
        or execution_manifest.get("read_human_gold") is not False
        or execution_manifest.get("applied_assignments") is not False
        or execution_manifest.get("created_or_mutated_identities") is not False
        or execution_manifest.get("mutated_profiles_or_references") is not False
        or execution_manifest.get("wrote_external_provider") is not False
        or execution_manifest.get("enabled_default_integration") is not False
        or execution_manifest.get("ran_historical_reprocessing") is not False
    ):
        raise Plan0057AuditError("Execution safety evidence is incomplete.")
    source_results = execution_manifest.get("source_results")
    stop_reasons = execution_manifest.get("stop_reasons")
    if (
        not isinstance(source_results, list)
        or len(source_results) != 3
        or not isinstance(stop_reasons, list)
    ):
        raise Plan0057AuditError("Recording execution evidence is incomplete.")
    rows: list[dict[str, Any]] = []
    normalized_stops: list[dict[str, str]] = []
    seen_cards: set[str] = set()
    for source in source_results:
        if not isinstance(source, Mapping):
            raise Plan0057AuditError("A source execution row is invalid.")
        document_id = str(source.get("document_id") or "")
        conversation_key = str(source.get("conversation_key") or "")
        proposals = source.get("proposals")
        if not isinstance(proposals, list) or not document_id or not conversation_key:
            raise Plan0057AuditError("A source execution row is incomplete.")
        stop_reason = source.get("stop_reason")
        if stop_reason:
            normalized_stops.append(
                {"document_id": document_id, "reason": str(stop_reason)}
            )
        if (
            source.get("eligible_speaker_count") != len(proposals)
            or source.get("covered_speaker_count") != len(proposals)
        ):
            raise Plan0057AuditError("A source speaker denominator is inconsistent.")
        for proposal in proposals:
            if not isinstance(proposal, Mapping):
                raise Plan0057AuditError("An acoustic proposal is invalid.")
            speaker_ref = str(proposal.get("speaker_ref") or "")
            card_id = f"{document_id}::{speaker_ref}"
            if (
                card_id in seen_cards
                or not _valid_audit_proposal(proposal)
            ):
                raise Plan0057AuditError("An acoustic proposal is unbound or unsafe.")
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
    if normalized_stops != stop_reasons:
        raise Plan0057AuditError("Stop-reason evidence is inconsistent.")
    return rows, normalized_stops


def recompute_plan0057_audit(
    *,
    execution_manifest: Mapping[str, Any],
    review_preview: Mapping[str, Any],
    current_identity_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Independently recompute yield, correctness, burden, and safety."""

    execution_rows, stop_reasons = _indexed_execution_rows(execution_manifest)
    decisions = review_preview.get("decisions")
    identity_before = execution_manifest.get("identity_state_before")
    identity_after = execution_manifest.get("identity_state_after")
    if (
        review_preview.get("schema_version") != review.REVIEW_PREVIEW_SCHEMA
        or review_preview.get("status") != "complete_human_review_ready_to_freeze"
        or review_preview.get("execution_content_sha256")
        != review.EXECUTION_CONTENT_SHA256
        or review_preview.get("review_complete") is not True
        or review_preview.get("decision_count") != 15
        or review_preview.get("action_vector") != review.REVIEW_ACTION_VECTOR
        or not isinstance(decisions, list)
        or len(decisions) != 15
        or identity_before != identity_after
        or dict(current_identity_state) != identity_after
    ):
        raise Plan0057AuditError("Review or identity-state guard is incomplete.")
    execution_by_card = {row["card_id"]: row for row in execution_rows}
    decisions_by_card = {
        str(item.get("card_id") or ""): item
        for item in decisions
        if isinstance(item, Mapping)
    }
    if (
        len(execution_rows) != 15
        or len(execution_by_card) != 15
        or len(decisions_by_card) != 15
        or execution_by_card.keys() != decisions_by_card.keys()
    ):
        raise Plan0057AuditError("Proposal and human-review denominators differ.")

    rows = []
    for execution_row in execution_rows:
        card_id = execution_row["card_id"]
        proposal = execution_row["proposal"]
        decision = decisions_by_card[card_id]
        proposed_subject = proposal.get("subject_id")
        actual_identity = decision.get("actual_identity")
        display_label = decision.get("review_display_label")
        if (
            decision.get("document_id") != execution_row["document_id"]
            or decision.get("conversation_key") != execution_row["conversation_key"]
            or decision.get("speaker_ref") != execution_row["speaker_ref"]
            or decision.get("proposed_subject_id") != proposed_subject
            or actual_identity
            not in ALLOWLISTED_SUBJECT_IDS | NON_ENROLLED_IDENTITIES
            or (
                display_label is not None
                and (
                    actual_identity != "neither_enrolled"
                    or not isinstance(display_label, str)
                    or not display_label.strip()
                    or len(display_label) > 120
                )
            )
        ):
            raise Plan0057AuditError("A review decision is invalid or unbound.")
        if proposed_subject is None:
            expected_decision = (
                "confirm_abstention"
                if actual_identity == "neither_enrolled"
                else "reject_abstention"
            )
        else:
            expected_decision = (
                "confirm" if actual_identity == proposed_subject else "reject"
            )
        if decision.get("proposal_decision") != expected_decision:
            raise Plan0057AuditError("A review decision is internally inconsistent.")
        is_unknown = actual_identity == "unknown"
        is_enrolled = actual_identity in ALLOWLISTED_SUBJECT_IDS
        proposal_confirmed = (
            proposed_subject is not None and proposed_subject == actual_identity
        )
        abstention_correct = (
            proposed_subject is None and actual_identity == "neither_enrolled"
        )
        correct_disposition = proposal_confirmed or abstention_correct
        wrong_disposition = not is_unknown and not correct_disposition
        assignment = proposal.get("disposition") == "assign"
        rows.append(
            {
                "card_id": card_id,
                "document_id": execution_row["document_id"],
                "speaker_ref": execution_row["speaker_ref"],
                "disposition": proposal.get("disposition"),
                "confidence_band": proposal.get("confidence_band"),
                "proposed_subject_id": proposed_subject,
                "actual_identity": actual_identity,
                "review_display_label": display_label,
                "proposal_confirmed": proposal_confirmed,
                "abstention_correct": abstention_correct,
                "correct_proposal_disposition": correct_disposition,
                "wrong_proposal_disposition": wrong_disposition,
                "high_confidence_wrong": (
                    wrong_disposition and proposal.get("confidence_band") == "high"
                ),
                "correct_assignment": assignment and correct_disposition,
                "wrong_assignment": assignment and wrong_disposition,
                "is_enrolled": is_enrolled,
                "is_unknown": is_unknown,
            }
        )

    speaker_count = len(rows)
    enrolled_rows = [row for row in rows if row["is_enrolled"]]
    proposal_rows = [row for row in rows if row["proposed_subject_id"] is not None]
    confirmed_count = sum(row["proposal_confirmed"] for row in proposal_rows)
    correct_enrolled = sum(row["proposal_confirmed"] for row in enrolled_rows)
    metrics = {
        "eligible_recording_count": len(execution_manifest["source_results"]),
        "entered_recording_count": sum(
            source.get("entered") is True
            for source in execution_manifest["source_results"]
        ),
        "stop_reason_count": len(stop_reasons),
        "eligible_speaker_count": sum(
            int(source["eligible_speaker_count"])
            for source in execution_manifest["source_results"]
        ),
        "covered_speaker_count": speaker_count,
        "human_review_decision_count": len(decisions),
        "proposal_count": len(proposal_rows),
        "proposal_confirmed_count": confirmed_count,
        "proposal_rejected_count": len(proposal_rows) - confirmed_count,
        "assign_disposition_count": sum(
            row["disposition"] == "assign" for row in rows
        ),
        "review_disposition_count": sum(
            row["disposition"] == "review" for row in rows
        ),
        "abstention_count": sum(row["disposition"] == "abstain" for row in rows),
        "correct_abstention_count": sum(row["abstention_correct"] for row in rows),
        "correct_proposal_disposition_count": sum(
            row["correct_proposal_disposition"] for row in rows
        ),
        "wrong_proposal_disposition_count": sum(
            row["wrong_proposal_disposition"] for row in rows
        ),
        "high_confidence_wrong_count": sum(
            row["high_confidence_wrong"] for row in rows
        ),
        "correct_assignment_count": sum(row["correct_assignment"] for row in rows),
        "wrong_assignment_count": sum(row["wrong_assignment"] for row in rows),
        "unknown_identity_count": sum(row["is_unknown"] for row in rows),
        "enrolled_speaker_count": len(enrolled_rows),
        "enrolled_correct_proposal_count": correct_enrolled,
        "enrolled_recall": (
            correct_enrolled / len(enrolled_rows) if enrolled_rows else 1.0
        ),
        "proposal_precision": (
            confirmed_count / len(proposal_rows) if proposal_rows else 1.0
        ),
        "review_burden": len(decisions) / speaker_count if speaker_count else 0.0,
        "manual_resolution_burden": (
            sum(row["disposition"] != "assign" for row in rows) / speaker_count
            if speaker_count
            else 0.0
        ),
        "identity_creation_count": 0,
        "profile_or_reference_mutation_count": 0,
        "speaker_assignment_write_count": 0,
        "provider_write_count": 0,
    }
    if (
        metrics["eligible_recording_count"] != 3
        or metrics["entered_recording_count"] != 3
        or metrics["eligible_speaker_count"] != 15
        or metrics["covered_speaker_count"] != 15
        or metrics["human_review_decision_count"] != 15
        or metrics["stop_reason_count"] != 0
    ):
        raise Plan0057AuditError("The terminal denominator is incomplete.")
    if metrics["high_confidence_wrong_count"]:
        terminal_decision = "stop"
    elif (
        metrics["wrong_proposal_disposition_count"]
        or metrics["enrolled_recall"] < 1.0
        or metrics["unknown_identity_count"]
    ):
        terminal_decision = "refine"
    else:
        terminal_decision = "plan_next_bounded_milestone"
    core = {
        "schema_version": AUDIT_SCHEMA,
        "status": "independent_recomputation_complete",
        "execution_content_sha256": review.EXECUTION_CONTENT_SHA256,
        "review_content_sha256": review_preview["content_sha256"],
        "rows": rows,
        "metrics": metrics,
        "stop_reasons": stop_reasons,
        "identity_state_before": dict(identity_before),
        "identity_state_after": dict(identity_after),
        "identity_state_current": dict(current_identity_state),
        "identity_state_unchanged": True,
        "terminal_decision": terminal_decision,
        "independent_guard_recomputed": True,
        "action_vector": dict(AUDIT_ACTION_VECTOR),
    }
    return {**core, "content_sha256": canonical_hash(core)}


def preview_plan0057_terminal_audit(
    *,
    execution_manifest: Mapping[str, Any],
    review_preview: Mapping[str, Any],
    current_identity_state: Mapping[str, Any],
    repository_authority: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        repository_authority.get("clean") is not True
        or repository_authority.get("upstream_ahead") != 0
        or repository_authority.get("upstream_behind") != 0
    ):
        raise Plan0057AuditError("Repository authority must be clean and upstream-even.")
    audit = recompute_plan0057_audit(
        execution_manifest=execution_manifest,
        review_preview=review_preview,
        current_identity_state=current_identity_state,
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
    return {**core, "content_sha256": canonical_hash(core)}


def _terminal_paths(runtime_root: Path, content_sha256: str) -> dict[str, Path]:
    root = runtime_root.expanduser().absolute()
    run = root / f"terminal-audit-{content_sha256[:24]}"
    return {
        "root": root,
        "run": run,
        "manifest": run / "private-manifest.json",
        "receipt": run / "receipt.json",
    }


def _terminal_receipt(
    preview: Mapping[str, Any],
    manifest_sha256: str,
) -> dict[str, Any]:
    metrics = preview["metrics"]
    return {
        "schema_version": TERMINAL_RECEIPT_SCHEMA,
        "status": "terminal_decision_frozen",
        "preview_content_sha256": preview["content_sha256"],
        "manifest_sha256": manifest_sha256,
        "audit_content_sha256": preview["audit_content_sha256"],
        "terminal_decision": preview["terminal_decision"],
        "eligible_recording_count": metrics["eligible_recording_count"],
        "entered_recording_count": metrics["entered_recording_count"],
        "eligible_speaker_count": metrics["eligible_speaker_count"],
        "covered_speaker_count": metrics["covered_speaker_count"],
        "human_review_decision_count": metrics["human_review_decision_count"],
        "wrong_proposal_disposition_count": metrics[
            "wrong_proposal_disposition_count"
        ],
        "high_confidence_wrong_count": metrics["high_confidence_wrong_count"],
        "enrolled_recall": metrics["enrolled_recall"],
        "proposal_precision": metrics["proposal_precision"],
        "identity_state_unchanged": True,
        "applied_assignments": False,
        "mode": "0600",
    }


def freeze_plan0057_terminal_audit(
    reviewed_preview: Mapping[str, Any],
    *,
    expected_content_sha256: str,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
) -> dict[str, Any]:
    preview = dict(reviewed_preview)
    core = {key: value for key, value in preview.items() if key != "content_sha256"}
    if (
        preview.get("schema_version") != TERMINAL_PREVIEW_SCHEMA
        or preview.get("content_sha256") != expected_content_sha256
        or canonical_hash(core) != expected_content_sha256
        or preview.get("applied_assignments") is not False
        or preview.get("action_vector") != AUDIT_ACTION_VECTOR
    ):
        raise Plan0057AuditError("Reviewed terminal audit is stale or unsafe.")
    paths = _terminal_paths(runtime_root, expected_content_sha256)
    if paths["receipt"].exists():
        return replay_plan0057_terminal_audit(
            expected_content_sha256,
            runtime_root=runtime_root,
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


def replay_plan0057_terminal_audit(
    expected_content_sha256: str,
    *,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
) -> dict[str, Any]:
    paths = _terminal_paths(runtime_root, expected_content_sha256)
    require_private_file(paths["manifest"], paths["root"])
    require_private_file(paths["receipt"], paths["root"])
    manifest = read_private_object(paths["manifest"])
    receipt = read_private_object(paths["receipt"])
    preview = manifest.get("preview")
    if not isinstance(preview, Mapping):
        raise Plan0057AuditError("Frozen terminal audit is invalid.")
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
        or canonical_hash(core) != expected_content_sha256
        or preview.get("audit", {}).get("content_sha256")
        != preview.get("audit_content_sha256")
        or preview.get("action_vector") != AUDIT_ACTION_VECTOR
    ):
        raise Plan0057AuditError("Frozen terminal audit evidence drifted.")
    return {
        **receipt,
        "replay_schema_version": TERMINAL_REPLAY_SCHEMA,
        "idempotent_replay": True,
    }


def build_live_terminal_audit_preview(
    review_content_sha256: str,
) -> dict[str, Any]:
    replay = review.replay_plan0057_review(review_content_sha256)
    review_paths = review._review_paths(review.DEFAULT_RUNTIME_ROOT, review_content_sha256)
    require_private_file(review_paths["manifest"], review_paths["root"])
    review_manifest = read_private_object(review_paths["manifest"])
    review_preview = review_manifest.get("preview")
    execution_manifest = review._live_execution_manifest()
    if (
        replay.get("idempotent_replay") is not True
        or not isinstance(review_preview, Mapping)
    ):
        raise Plan0057AuditError("Frozen review evidence is invalid.")
    return preview_plan0057_terminal_audit(
        execution_manifest=execution_manifest,
        review_preview=review_preview,
        current_identity_state=execution._current_identity_state(),
        repository_authority=_repository_authority(),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Freeze or replay Plan 0057 terminal audit.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("preview", "freeze"):
        child = subparsers.add_parser(command)
        child.add_argument("--review-content-sha256", required=True)
        child.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
        if command == "freeze":
            child.add_argument("--expected-content-sha256", required=True)
    replay = subparsers.add_parser("replay")
    replay.add_argument("--audit-content-sha256", required=True)
    replay.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "replay":
        result = replay_plan0057_terminal_audit(
            args.audit_content_sha256,
            runtime_root=args.runtime_root,
        )
    else:
        preview = build_live_terminal_audit_preview(args.review_content_sha256)
        if args.command == "preview":
            result = preview
        else:
            result = freeze_plan0057_terminal_audit(
                preview,
                expected_content_sha256=args.expected_content_sha256,
                runtime_root=args.runtime_root,
            )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
