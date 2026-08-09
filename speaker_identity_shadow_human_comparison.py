from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import subprocess
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import speaker_identity_shadow_human_review as review
from acoustic_audio_derivatives import (
    canonical_artifact_hash,
    ensure_private_tree,
    read_private_object,
    require_private_file,
    sha256_file,
    write_immutable_private_json,
)


DECISION_MANIFEST_SCHEMA = "transcribe-audio.plan0061-human-gold-manifest.v1"
DECISION_RECEIPT_SCHEMA = "transcribe-audio.plan0061-human-gold-receipt.v1"
COMPARISON_SCHEMA = "transcribe-audio.plan0061-three-condition-comparison.v1"
COMPARISON_RECEIPT_SCHEMA = "transcribe-audio.plan0061-comparison-receipt.v1"
TERMINAL_AUDIT_SCHEMA = "transcribe-audio.plan0061-terminal-audit.v1"
MODULE_PATH = Path(__file__).name
DEFAULT_RUNTIME_ROOT = review.DEFAULT_RUNTIME_ROOT
HIGH_CONFIDENCE_THRESHOLD = 0.75


class Plan0061ComparisonError(ValueError):
    """Raised when human gold or comparison evidence cannot be frozen exactly."""


def _fail(message: str) -> None:
    raise Plan0061ComparisonError(message)


def _git(arguments: Sequence[str], *, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", *arguments],
        cwd=Path(__file__).resolve().parent,
        capture_output=True,
        text=not binary,
        check=False,
    )
    if result.returncode:
        _fail("Repository authority is unavailable.")
    return result.stdout if binary else result.stdout.strip()


def _repository_authority() -> dict[str, Any]:
    if _git(["status", "--porcelain=v1", "--untracked-files=normal"]):
        _fail("Repository must be clean before freezing human gold.")
    if str(_git(["rev-list", "--left-right", "--count", "@{upstream}...HEAD"])).split() != [
        "0",
        "0",
    ]:
        _fail("Repository must be upstream-even before freezing human gold.")
    commit = str(_git(["rev-parse", "HEAD"]))
    modules = {}
    for module_path in (MODULE_PATH, review.MODULE_PATH):
        committed = _git(["show", f"{commit}:{module_path}"], binary=True)
        current_path = Path(__file__).resolve().parent / module_path
        current_sha256 = sha256_file(current_path)
        if (
            not isinstance(committed, bytes)
            or hashlib.sha256(committed).hexdigest() != current_sha256
        ):
            _fail("Committed Plan 0061 comparison authority drifted.")
        modules[module_path] = current_sha256
    return {
        "commit": commit,
        "modules": modules,
        "clean": True,
        "upstream_ahead": 0,
        "upstream_behind": 0,
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def _with_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    core = dict(payload)
    return {**core, "content_sha256": canonical_artifact_hash(core)}


def _has_valid_content_hash(payload: Mapping[str, Any]) -> bool:
    return payload.get("content_sha256") == canonical_artifact_hash(
        {key: value for key, value in payload.items() if key != "content_sha256"}
    )


def _validated_submission(
    submission: Mapping[str, Any], manifest: Mapping[str, Any]
) -> tuple[list[dict[str, str]], dict[str, set[str]]]:
    cases = review.normalized_review_cases(manifest)
    expected_slots: list[str] = []
    allowed_by_slot: dict[str, set[str]] = {}
    for case in cases:
        for slot in case["slots"]:
            expected_slots.append(slot["slot_id"])
            allowed_by_slot[slot["slot_id"]] = set(slot["allowed_decisions"])
    decisions = submission.get("decisions")
    if not isinstance(decisions, list):
        _fail("The human-gold decisions are unavailable.")
    normalized = []
    for item in decisions:
        if not isinstance(item, Mapping):
            _fail("A human-gold decision is invalid.")
        slot_id = str(item.get("slot_id") or "")
        decision = str(item.get("decision") or "")
        normalized.append({"slot_id": slot_id, "decision": decision})
    core = {key: value for key, value in submission.items() if key != "content_sha256"}
    if (
        submission.get("schema_version") != review.DECISION_SUBMISSION_SCHEMA
        or submission.get("status") != "complete_operator_decisions_preview"
        or submission.get("plan0060_activation_sha256")
        != review.PLAN0060_ACTIVATION_SHA256
        or submission.get("p4_content_sha256") != review.PLAN0060_P4_CONTENT_SHA256
        or submission.get("p4_manifest_sha256") != review.PLAN0060_P4_MANIFEST_SHA256
        or submission.get("decision_count") != review.EXPECTED_SPEAKERS
        or [item["slot_id"] for item in normalized] != expected_slots
        or any(
            item["decision"] not in allowed_by_slot.get(item["slot_id"], set())
            for item in normalized
        )
        or any(submission.get(field) is not False for field in (
            "applied_assignments",
            "created_or_mutated_identities",
            "mutated_profiles_or_references",
            "wrote_live_knowledge",
            "wrote_external_provider",
            "wrote_graphiti",
        ))
        or submission.get("content_sha256") != canonical_artifact_hash(core)
    ):
        _fail("The complete human-gold submission drifted.")
    return normalized, allowed_by_slot


def recompute_comparison(
    manifest: Mapping[str, Any], submission: Mapping[str, Any]
) -> dict[str, Any]:
    """Independently score all three frozen conditions against exact human gold."""

    decisions, allowed_by_slot = _validated_submission(submission, manifest)
    decision_by_slot = {item["slot_id"]: item["decision"] for item in decisions}
    raw_slots: dict[str, tuple[str, Mapping[str, Any]]] = {}
    candidate_ids_by_document: dict[str, set[str]] = {}
    case_failure_count = 0
    for case in manifest.get("cases") or []:
        if not isinstance(case, Mapping):
            _fail("A comparison case is invalid.")
        document_id = str(case.get("document_id") or "")
        candidate_ids_by_document[document_id] = {
            str(item.get("person_id") or "")
            for item in case.get("candidate_options") or []
            if isinstance(item, Mapping)
        }
        failures = case.get("source_failures")
        if not isinstance(failures, list):
            _fail("A comparison case has invalid source failures.")
        case_failure_count += len(failures)
        for slot in case.get("speaker_slots") or []:
            if not isinstance(slot, Mapping):
                _fail("A comparison speaker slot is invalid.")
            speaker_ref = str(slot.get("speaker_ref") or "")
            slot_id = f"{document_id}::{speaker_ref}"
            if slot_id in raw_slots:
                _fail("The comparison source repeats a speaker slot.")
            raw_slots[slot_id] = (document_id, slot)
    if set(raw_slots) != set(decision_by_slot):
        _fail("The comparison and human-gold denominators differ.")

    gold_counts = Counter()
    candidate_recalled_count = 0
    rows: list[dict[str, Any]] = []
    condition_counters = {
        condition: Counter(
            {
                "evaluation_count": 0,
                "proposal_count": 0,
                "correct_proposal_count": 0,
                "wrong_proposal_count": 0,
                "high_confidence_wrong_count": 0,
                "known_person_count": 0,
                "known_person_recalled_count": 0,
                "not_listed_count": 0,
                "unresolved_count": 0,
                "appropriate_abstention_count": 0,
                "inappropriate_abstention_count": 0,
                "provenance_complete_count": 0,
                "evaluation_source_failure_count": 0,
            }
        )
        for condition in review.CONDITIONS
    }
    proposed_by_condition_document: dict[str, dict[str, list[str]]] = {
        condition: defaultdict(list) for condition in review.CONDITIONS
    }

    for decision in decisions:
        slot_id = decision["slot_id"]
        selected = decision["decision"]
        document_id, raw_slot = raw_slots[slot_id]
        candidates = candidate_ids_by_document[document_id]
        if selected == "not_listed":
            gold_outcome = "not_listed"
            gold_counts["not_listed"] += 1
        elif selected == "unresolved":
            gold_outcome = "unresolved"
            gold_counts["unresolved"] += 1
        else:
            gold_outcome = "person"
            gold_counts["person"] += 1
            candidate_recalled_count += int(selected in candidates)
        condition_rows = []
        raw_conditions = raw_slot.get("conditions")
        if not isinstance(raw_conditions, list) or len(raw_conditions) != len(
            review.CONDITIONS
        ):
            _fail("A comparison slot has an invalid condition denominator.")
        for raw_condition in raw_conditions:
            if not isinstance(raw_condition, Mapping):
                _fail("A comparison condition is invalid.")
            condition = str(raw_condition.get("condition") or "")
            if condition not in condition_counters:
                _fail("A comparison condition is unknown.")
            counter = condition_counters[condition]
            counter["evaluation_count"] += 1
            if gold_outcome == "person":
                counter["known_person_count"] += 1
            else:
                counter[f"{gold_outcome}_count"] += 1
            proposed = raw_condition.get("proposed_person_id")
            if proposed is not None:
                proposed = str(proposed)
                if proposed not in candidates:
                    _fail("A comparison proposal escapes the frozen candidate set.")
                proposed_by_condition_document[condition][document_id].append(proposed)
            correct = proposed is not None and gold_outcome == "person" and proposed == selected
            wrong = proposed is not None and not correct
            abstained = proposed is None and raw_condition.get("outcome") == "abstained"
            appropriate_abstention = abstained and gold_outcome in {
                "not_listed",
                "unresolved",
            }
            inappropriate_abstention = abstained and gold_outcome == "person"
            counter["proposal_count"] += int(proposed is not None)
            counter["correct_proposal_count"] += int(correct)
            counter["wrong_proposal_count"] += int(wrong)
            counter["high_confidence_wrong_count"] += int(
                wrong
                and float(raw_condition.get("capped_confidence") or 0.0)
                >= HIGH_CONFIDENCE_THRESHOLD
            )
            counter["known_person_recalled_count"] += int(correct)
            counter["appropriate_abstention_count"] += int(appropriate_abstention)
            counter["inappropriate_abstention_count"] += int(inappropriate_abstention)
            factors = raw_condition.get("factors")
            failures = raw_condition.get("source_failures")
            if not isinstance(factors, list) or not isinstance(failures, list):
                _fail("A comparison condition has invalid provenance.")
            provenance_complete = bool(factors) and all(
                isinstance(factor, Mapping)
                and bool(factor.get("evidence_ids"))
                and bool(factor.get("independence_groups"))
                for factor in factors
            )
            counter["provenance_complete_count"] += int(provenance_complete)
            counter["evaluation_source_failure_count"] += len(failures)
            condition_rows.append(
                {
                    "condition": condition,
                    "evaluation_id": raw_condition.get("evaluation_id"),
                    "outcome": raw_condition.get("outcome"),
                    "proposed_person_id": proposed,
                    "correct_proposal": correct,
                    "wrong_proposal": wrong,
                    "high_confidence_wrong": (
                        wrong
                        and float(raw_condition.get("capped_confidence") or 0.0)
                        >= HIGH_CONFIDENCE_THRESHOLD
                    ),
                    "appropriate_abstention": appropriate_abstention,
                    "inappropriate_abstention": inappropriate_abstention,
                    "provenance_complete": provenance_complete,
                    "source_failure_count": len(failures),
                }
            )
        rows.append(
            {
                "slot_id": slot_id,
                "gold_outcome": gold_outcome,
                "gold_person_id": selected if gold_outcome == "person" else None,
                "candidate_present": selected in allowed_by_slot[slot_id]
                and gold_outcome == "person",
                "conditions": condition_rows,
            }
        )

    condition_metrics: dict[str, dict[str, Any]] = {}
    for condition, counter in condition_counters.items():
        duplicate_forks = sum(
            sum(count - 1 for count in Counter(proposals).values() if count > 1)
            for proposals in proposed_by_condition_document[condition].values()
        )
        metrics = dict(counter)
        metrics.update(
            {
                "top_person_correctness": _ratio(
                    counter["correct_proposal_count"], counter["known_person_count"]
                ),
                "enrolled_recall": _ratio(
                    counter["known_person_recalled_count"], counter["known_person_count"]
                ),
                "precision": _ratio(
                    counter["correct_proposal_count"], counter["proposal_count"]
                ),
                "appropriate_abstention_rate": _ratio(
                    counter["appropriate_abstention_count"],
                    counter["not_listed_count"] + counter["unresolved_count"],
                ),
                "provenance_completeness": _ratio(
                    counter["provenance_complete_count"], counter["evaluation_count"]
                ),
                "duplicate_person_fork_count": duplicate_forks,
                "provider_failure_count": (
                    counter["evaluation_source_failure_count"] + case_failure_count
                ),
            }
        )
        condition_metrics[condition] = metrics

    def _delta(left: str, right: str, metric: str) -> float | None:
        left_value = condition_metrics[left][metric]
        right_value = condition_metrics[right][metric]
        if left_value is None or right_value is None:
            return None
        return round(float(left_value) - float(right_value), 4)

    candidate_recall = _ratio(candidate_recalled_count, review.EXPECTED_SPEAKERS)
    unresolved_rate = _ratio(gold_counts["unresolved"], review.EXPECTED_SPEAKERS)
    all_high_confidence_wrong = sum(
        metrics["high_confidence_wrong_count"] for metrics in condition_metrics.values()
    )
    terminal_decision = "stop" if all_high_confidence_wrong else "complete"
    if terminal_decision == "complete" and (
        candidate_recall != 1.0
        or any(metrics["enrolled_recall"] != 1.0 for metrics in condition_metrics.values())
    ):
        terminal_decision = "refine"
    core = {
        "schema_version": COMPARISON_SCHEMA,
        "status": "comparison_complete",
        "submission_content_sha256": submission["content_sha256"],
        "p4_content_sha256": review.PLAN0060_P4_CONTENT_SHA256,
        "p4_manifest_sha256": review.PLAN0060_P4_MANIFEST_SHA256,
        "recording_count": review.EXPECTED_RECORDINGS,
        "speaker_slot_count": review.EXPECTED_SPEAKERS,
        "condition_count": review.EXPECTED_CONDITIONS,
        "gold_metrics": {
            "person_count": gold_counts["person"],
            "not_listed_count": gold_counts["not_listed"],
            "unresolved_count": gold_counts["unresolved"],
            "candidate_recalled_count": candidate_recalled_count,
            "candidate_recall_denominator": review.EXPECTED_SPEAKERS,
            "candidate_recall": candidate_recall,
            "unresolved_rate": unresolved_rate,
        },
        "condition_metrics": condition_metrics,
        "condition_deltas": {
            "acoustic_only_minus_context_only": {
                metric: _delta("acoustic_only", "context_only", metric)
                for metric in (
                    "top_person_correctness",
                    "enrolled_recall",
                    "appropriate_abstention_rate",
                )
            },
            "combined_minus_context_only": {
                metric: _delta("combined", "context_only", metric)
                for metric in (
                    "top_person_correctness",
                    "enrolled_recall",
                    "appropriate_abstention_rate",
                )
            },
        },
        "review_burden": {
            "human_decision_count": review.EXPECTED_SPEAKERS,
            "recording_count": review.EXPECTED_RECORDINGS,
            "condition_view_count": review.EXPECTED_CONDITIONS,
            "audio_clip_count": review.EXPECTED_SPEAKERS,
            "not_listed_followup_count": gold_counts["not_listed"],
        },
        "high_confidence_threshold": HIGH_CONFIDENCE_THRESHOLD,
        "terminal_decision": terminal_decision,
        "rows": rows,
        "negative_actions": review.NEGATIVE_ACTIONS,
    }
    return _with_hash(core)


def _paths(runtime_root: Path, submission_sha256: str) -> dict[str, Path]:
    root = runtime_root.expanduser().absolute()
    run = root / f"human-gold-{submission_sha256[:24]}"
    return {
        "root": root,
        "run": run,
        "decision": run / "private-human-gold.json",
        "decision_receipt": run / "decision-receipt.json",
        "comparison": run / "private-comparison.json",
        "comparison_receipt": run / "comparison-receipt.json",
        "terminal": run / "terminal-audit.json",
    }


def _safe_result(paths: Mapping[str, Path], comparison: Mapping[str, Any], *, replay: bool) -> dict[str, Any]:
    return {
        "status": "plan0061_complete",
        "submission_content_sha256": comparison["submission_content_sha256"],
        "comparison_content_sha256": comparison["content_sha256"],
        "terminal_decision": comparison["terminal_decision"],
        "recording_count": comparison["recording_count"],
        "speaker_slot_count": comparison["speaker_slot_count"],
        "condition_count": comparison["condition_count"],
        "gold_metrics": comparison["gold_metrics"],
        "condition_metrics": comparison["condition_metrics"],
        "condition_deltas": comparison["condition_deltas"],
        "review_burden": comparison["review_burden"],
        "decision_manifest_sha256": sha256_file(paths["decision"]),
        "comparison_manifest_sha256": sha256_file(paths["comparison"]),
        "terminal_manifest_sha256": sha256_file(paths["terminal"]),
        "runtime_path": str(paths["run"]),
        "idempotent_replay": replay,
        "live_mutation_count": 0,
    }


def freeze_human_gold_and_comparison(
    answer_text: str,
    *,
    plan0060_root: Path = review.DEFAULT_PLAN0060_ROOT,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    live_store_root: Path = review.DEFAULT_LIVE_STORE_ROOT,
) -> dict[str, Any]:
    """Freeze exact P3 gold, P4 comparison, and P5 audit without applying it."""

    repository = _repository_authority()
    source, bindings = review._validated_live_source(
        plan0060_root=plan0060_root, live_store_root=live_store_root
    )
    submission = review.parse_decision_block(answer_text, source)
    comparison = recompute_comparison(source, submission)
    paths = _paths(runtime_root, submission["content_sha256"])
    if paths["terminal"].exists():
        return replay_human_gold_and_comparison(
            submission["content_sha256"],
            plan0060_root=plan0060_root,
            runtime_root=runtime_root,
            live_store_root=live_store_root,
        )
    if paths["run"].exists():
        _fail("A partial Plan 0061 human-gold directory already exists.")
    ensure_private_tree(paths["root"], paths["run"])
    decision_manifest = _with_hash(
        {
            "schema_version": DECISION_MANIFEST_SCHEMA,
            "status": "human_gold_frozen",
            "submission": submission,
            "source_bindings": bindings,
            "repository_authority": repository,
            "decision_count": review.EXPECTED_SPEAKERS,
            "negative_actions": review.NEGATIVE_ACTIONS,
        }
    )
    write_immutable_private_json(paths["decision"], decision_manifest)
    decision_receipt = _with_hash(
        {
            "schema_version": DECISION_RECEIPT_SCHEMA,
            "status": "human_gold_frozen",
            "submission_content_sha256": submission["content_sha256"],
            "decision_manifest_sha256": sha256_file(paths["decision"]),
            "decision_count": review.EXPECTED_SPEAKERS,
            "live_mutation_count": 0,
        }
    )
    write_immutable_private_json(paths["decision_receipt"], decision_receipt)
    write_immutable_private_json(paths["comparison"], comparison)
    comparison_receipt = _with_hash(
        {
            "schema_version": COMPARISON_RECEIPT_SCHEMA,
            "status": "comparison_frozen",
            "submission_content_sha256": submission["content_sha256"],
            "comparison_content_sha256": comparison["content_sha256"],
            "comparison_manifest_sha256": sha256_file(paths["comparison"]),
            "terminal_decision": comparison["terminal_decision"],
            "live_mutation_count": 0,
        }
    )
    write_immutable_private_json(paths["comparison_receipt"], comparison_receipt)
    terminal = _with_hash(
        {
            "schema_version": TERMINAL_AUDIT_SCHEMA,
            "status": "complete",
            "terminal_decision": comparison["terminal_decision"],
            "submission_content_sha256": submission["content_sha256"],
            "comparison_content_sha256": comparison["content_sha256"],
            "decision_manifest_sha256": sha256_file(paths["decision"]),
            "comparison_manifest_sha256": sha256_file(paths["comparison"]),
            "metrics_recomputed": recompute_comparison(source, submission) == comparison,
            "decision_reasons": (
                ["high_confidence_wrong_proposal"]
                if comparison["terminal_decision"] == "stop"
                else (
                    [
                        "candidate_pool_incomplete",
                        "known_person_recall_incomplete",
                        "no_condition_improved_known_person_recall",
                    ]
                    if comparison["terminal_decision"] == "refine"
                    else ["all_comparison_gates_passed"]
                )
            ),
            "source_bindings": bindings,
            "repository_authority": repository,
            "private_modes_required": {"directory": "0700", "file": "0600"},
            "live_mutation_count": 0,
            "negative_actions": review.NEGATIVE_ACTIONS,
        }
    )
    if terminal["metrics_recomputed"] is not True:
        _fail("The independent comparison recomputation disagreed.")
    write_immutable_private_json(paths["terminal"], terminal)
    return _safe_result(paths, comparison, replay=False)


def replay_human_gold_and_comparison(
    submission_sha256: str,
    *,
    plan0060_root: Path = review.DEFAULT_PLAN0060_ROOT,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    live_store_root: Path = review.DEFAULT_LIVE_STORE_ROOT,
) -> dict[str, Any]:
    selected_sha256 = review._sha256(submission_sha256, field="submission SHA-256")
    paths = _paths(runtime_root, selected_sha256)
    for key in (
        "decision",
        "decision_receipt",
        "comparison",
        "comparison_receipt",
        "terminal",
    ):
        require_private_file(paths[key], paths["root"])
    source, bindings = review._validated_live_source(
        plan0060_root=plan0060_root, live_store_root=live_store_root
    )
    current_repository = _repository_authority()
    decision = read_private_object(paths["decision"])
    submission = decision.get("submission")
    if not isinstance(submission, Mapping):
        _fail("The frozen human-gold submission is unavailable.")
    _validated_submission(submission, source)
    comparison = read_private_object(paths["comparison"])
    expected_comparison = recompute_comparison(source, submission)
    decision_receipt = read_private_object(paths["decision_receipt"])
    comparison_receipt = read_private_object(paths["comparison_receipt"])
    terminal = read_private_object(paths["terminal"])
    stored_repository = decision.get("repository_authority")
    if (
        selected_sha256 != submission.get("content_sha256")
        or decision.get("schema_version") != DECISION_MANIFEST_SCHEMA
        or decision.get("source_bindings") != bindings
        or decision.get("negative_actions") != review.NEGATIVE_ACTIONS
        or not _has_valid_content_hash(decision)
        or not isinstance(stored_repository, Mapping)
        or stored_repository.get("modules") != current_repository.get("modules")
        or comparison != expected_comparison
        or decision_receipt.get("schema_version") != DECISION_RECEIPT_SCHEMA
        or decision_receipt.get("decision_manifest_sha256") != sha256_file(paths["decision"])
        or not _has_valid_content_hash(decision_receipt)
        or comparison_receipt.get("schema_version") != COMPARISON_RECEIPT_SCHEMA
        or comparison_receipt.get("comparison_manifest_sha256")
        != sha256_file(paths["comparison"])
        or comparison_receipt.get("comparison_content_sha256")
        != comparison["content_sha256"]
        or not _has_valid_content_hash(comparison_receipt)
        or terminal.get("schema_version") != TERMINAL_AUDIT_SCHEMA
        or terminal.get("source_bindings") != bindings
        or terminal.get("repository_authority") != stored_repository
        or terminal.get("metrics_recomputed") is not True
        or terminal.get("comparison_content_sha256") != comparison["content_sha256"]
        or terminal.get("live_mutation_count") != 0
        or terminal.get("negative_actions") != review.NEGATIVE_ACTIONS
        or not _has_valid_content_hash(terminal)
    ):
        _fail("The frozen Plan 0061 comparison evidence drifted.")
    return _safe_result(paths, comparison, replay=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze or replay the non-applying Plan 0061 gold comparison."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--answers-file", type=Path, required=True)
    freeze.add_argument("--plan0060-root", type=Path, default=review.DEFAULT_PLAN0060_ROOT)
    freeze.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    freeze.add_argument("--live-store-root", type=Path, default=review.DEFAULT_LIVE_STORE_ROOT)
    replay = subparsers.add_parser("replay")
    replay.add_argument("--submission-sha256", required=True)
    replay.add_argument("--plan0060-root", type=Path, default=review.DEFAULT_PLAN0060_ROOT)
    replay.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    replay.add_argument("--live-store-root", type=Path, default=review.DEFAULT_LIVE_STORE_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "freeze":
            answer_path = args.answers_file.expanduser().absolute()
            if (
                not answer_path.is_file()
                or answer_path.is_symlink()
                or answer_path.stat().st_mode & 0o077
            ):
                _fail("The answer file must be a private 0600 regular file.")
            result = freeze_human_gold_and_comparison(
                answer_path.read_text(encoding="utf-8"),
                plan0060_root=args.plan0060_root,
                runtime_root=args.runtime_root,
                live_store_root=args.live_store_root,
            )
        else:
            result = replay_human_gold_and_comparison(
                args.submission_sha256,
                plan0060_root=args.plan0060_root,
                runtime_root=args.runtime_root,
                live_store_root=args.live_store_root,
            )
        print(json.dumps(result, sort_keys=True))
        return 0
    except (OSError, sqlite3.Error, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
