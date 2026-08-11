"""Plan 0064 P4 literal human-gold ingestion and measured shadow gate.

The module freezes a complete, authority-bound human decision export and
compares it with the four already-frozen P3 conditions.  It never applies a
speaker identity.  Automatic local acceptance remains withheld unless the
source-disjoint result and the separate reviewed-development replay both pass
non-vacuous quality gates.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

from acoustic_audio_derivatives import (
    canonical_artifact_hash,
    ensure_private_tree,
    read_private_object,
    require_private_file,
    sha256_file,
    write_immutable_private_json,
)
from speaker_identity_plan0064_p0 import DEFAULT_RUNTIME_ROOT
from speaker_identity_plan0064_p4_review import (
    ACTION_COUNTS,
    AUTHORITY_SCHEMA as CURRENT_REVIEW_AUTHORITY_SCHEMA,
    DECISION_SCHEMA,
    _authority_inputs as p4_authority_inputs,
    replay_p4_review,
)


GOLD_SCHEMA = "transcribe-audio.plan0064-p4-human-gold.v1"
GOLD_RECEIPT_SCHEMA = "transcribe-audio.plan0064-p4-human-gold-receipt.v1"
MEASUREMENT_SCHEMA = "transcribe-audio.plan0064-p4-measurement.v1"
MEASUREMENT_RECEIPT_SCHEMA = "transcribe-audio.plan0064-p4-measurement-receipt.v1"
TERMINAL_SCHEMA = "transcribe-audio.plan0064-p4-terminal.v1"
DEVELOPMENT_GATE_SCHEMA = "transcribe-audio.plan0064-development-replay-gate.v1"
LEGACY_REVIEW_AUTHORITY_SCHEMA = "transcribe-audio.plan0064-p4-review-authority.v1"
AUTHORITY_BRIDGE_SCHEMA = "transcribe-audio.plan0064-p4-review-authority-bridge.v1"
CONDITIONS = ("acoustic", "context", "combined", "residual_policy")
DECISIONS = ("canonical_person", "not_listed", "unresolved")
HIGH_SUPPORT_REASONS = {
    "pillar_agreement",
    "pillar_agreement_same_person_multi_label",
    "two_known_plus_one_independently_supported_residual",
}


class Plan0064P4MeasurementError(ValueError):
    """Raised when P4 human gold, measurement, or lineage is not exact."""


def _hash(value: Any) -> str:
    return canonical_artifact_hash(value)


def _content_addressed(value: Mapping[str, Any]) -> dict[str, Any]:
    core = dict(value)
    core.pop("content_sha256", None)
    return {**core, "content_sha256": _hash(core)}


def _read(path: Path) -> dict[str, Any]:
    value = read_private_object(path)
    if not isinstance(value, dict):
        raise Plan0064P4MeasurementError(f"Private artifact is not an object: {path}")
    return value


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _validate_content_addressed(value: Mapping[str, Any], *, label: str) -> None:
    core = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != _hash(core):
        raise Plan0064P4MeasurementError(f"{label} content hash drifted.")


def _authorities(
    p0_content_sha256: str, *, runtime_root: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    review_receipt = replay_p4_review(
        p0_content_sha256, runtime_root=runtime_root
    )
    review_root = Path(str(review_receipt["private_review_root"]))
    authority_path = review_root / "review-authority.json"
    require_private_file(authority_path, review_root)
    authority = _read(authority_path)
    _validate_content_addressed(authority, label="P4 review authority")

    _manifest, p3_receipt, resolution = p4_authority_inputs(
        p0_content_sha256, runtime_root=runtime_root
    )
    _validate_content_addressed(resolution, label="P3 resolution")
    if (
        review_receipt.get("authority_content_sha256")
        != authority.get("content_sha256")
        or authority.get("p3_receipt_content_sha256")
        != p3_receipt.get("content_sha256")
        or authority.get("p3_resolution_content_sha256")
        != resolution.get("content_sha256")
        or any((authority.get("action_counts") or {}).values())
        or any((resolution.get("action_counts") or {}).values())
    ):
        raise Plan0064P4MeasurementError("P4 review and P3 resolution bindings drifted.")
    return authority, p3_receipt, resolution


def bridge_human_gold_authority(
    submission: Mapping[str, Any],
    *,
    source_authority: Mapping[str, Any],
    target_authority: Mapping[str, Any],
    resolution: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Rebind literal decisions across the filename-only P4 authority revision."""

    _validate_content_addressed(source_authority, label="Source P4 review authority")
    _validate_content_addressed(target_authority, label="Target P4 review authority")
    if (
        source_authority.get("schema_version") != LEGACY_REVIEW_AUTHORITY_SCHEMA
        or target_authority.get("schema_version") != CURRENT_REVIEW_AUTHORITY_SCHEMA
        or set(submission)
        != {"schema_version", "authority_content_sha256", "decisions"}
        or submission.get("schema_version") != DECISION_SCHEMA
        or submission.get("authority_content_sha256")
        != source_authority.get("content_sha256")
    ):
        raise Plan0064P4MeasurementError(
            "The P4 authority bridge source is stale or malformed."
        )

    invariant_fields = (
        "status",
        "human_decision_count",
        "model_predictions_visible",
        "p3_receipt_content_sha256",
        "p3_resolution_content_sha256",
        "people",
        "action_counts",
    )
    if any(
        source_authority.get(key) != target_authority.get(key)
        for key in invariant_fields
    ):
        raise Plan0064P4MeasurementError(
            "The P4 authority bridge changes decision semantics or lineage."
        )
    if (
        target_authority.get("p3_resolution_content_sha256")
        != resolution.get("content_sha256")
        or any((source_authority.get("action_counts") or {}).values())
        or any((target_authority.get("action_counts") or {}).values())
    ):
        raise Plan0064P4MeasurementError(
            "The P4 authority bridge escapes the frozen zero-effect resolution."
        )

    source_cases = source_authority.get("cases")
    target_cases = target_authority.get("cases")
    if not isinstance(source_cases, list) or not isinstance(target_cases, list):
        raise Plan0064P4MeasurementError("The P4 authority bridge cases are malformed.")
    target_without_filenames = [
        {key: value for key, value in item.items() if key != "recording_filename"}
        if isinstance(item, Mapping)
        else item
        for item in target_cases
    ]
    if source_cases != target_without_filenames:
        raise Plan0064P4MeasurementError(
            "The P4 authority bridge changes cases beyond recording filenames."
        )

    filename_rows: list[dict[str, str]] = []
    filename_by_document: dict[str, str] = {}
    for case in target_cases:
        if not isinstance(case, Mapping):
            raise Plan0064P4MeasurementError("A target P4 review case is invalid.")
        document_id = str(case.get("document_id") or "")
        filename = str(case.get("recording_filename") or "")
        if (
            not document_id
            or not filename
            or Path(filename).name != filename
            or "/" in filename
            or "\\" in filename
        ):
            raise Plan0064P4MeasurementError(
                "A target P4 recording filename is invalid."
            )
        prior = filename_by_document.setdefault(document_id, filename)
        if prior != filename:
            raise Plan0064P4MeasurementError(
                "A target recording has inconsistent filenames."
            )
        if prior == filename and not any(
            row["document_id"] == document_id for row in filename_rows
        ):
            filename_rows.append(
                {"document_id": document_id, "recording_filename": filename}
            )
    if (
        len(filename_rows) != 12
        or target_authority.get("recording_filename_set_sha256")
        != _hash(filename_rows)
    ):
        raise Plan0064P4MeasurementError(
            "The target P4 recording filename set drifted."
        )

    source_gold = normalize_human_gold(
        submission, authority=source_authority, resolution=resolution
    )
    rebound = {
        "schema_version": submission["schema_version"],
        "authority_content_sha256": target_authority["content_sha256"],
        "decisions": submission["decisions"],
    }
    target_gold = normalize_human_gold(
        rebound, authority=target_authority, resolution=resolution
    )
    if source_gold["decisions"] != target_gold["decisions"]:
        raise Plan0064P4MeasurementError(
            "The P4 authority bridge changed literal human decisions."
        )
    bridge = _content_addressed(
        {
            "schema_version": AUTHORITY_BRIDGE_SCHEMA,
            "status": "exact_filename_only_authority_bridge",
            "source_authority_content_sha256": source_authority["content_sha256"],
            "target_authority_content_sha256": target_authority["content_sha256"],
            "source_preview_content_sha256": source_authority[
                "preview_content_sha256"
            ],
            "target_preview_content_sha256": target_authority[
                "preview_content_sha256"
            ],
            "p3_receipt_content_sha256": target_authority[
                "p3_receipt_content_sha256"
            ],
            "p3_resolution_content_sha256": target_authority[
                "p3_resolution_content_sha256"
            ],
            "recording_filename_set_sha256": target_authority[
                "recording_filename_set_sha256"
            ],
            "source_submission_sha256": _hash(dict(submission)),
            "rebound_submission_sha256": _hash(rebound),
            "decision_rows_sha256": _hash(source_gold["decisions"]),
            "decision_count": source_gold["decision_count"],
            "decision_type_counts": source_gold["decision_type_counts"],
            "changed_fields": ["authority_content_sha256"],
            "action_counts": dict(ACTION_COUNTS),
        }
    )
    return rebound, bridge


def normalize_human_gold(
    submission: Mapping[str, Any],
    *,
    authority: Mapping[str, Any],
    resolution: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one exact 39-row client export and freeze its literal meaning."""

    if set(submission) != {"schema_version", "authority_content_sha256", "decisions"}:
        raise Plan0064P4MeasurementError("The P4 decision export has unknown or missing fields.")
    if (
        submission.get("schema_version") != DECISION_SCHEMA
        or submission.get("authority_content_sha256")
        != authority.get("content_sha256")
    ):
        raise Plan0064P4MeasurementError("The P4 decision export is stale or misbound.")
    cases = authority.get("cases")
    people = authority.get("people")
    decisions = submission.get("decisions")
    if not isinstance(cases, list) or not isinstance(people, list):
        raise Plan0064P4MeasurementError("The P4 review authority is malformed.")
    if not isinstance(decisions, list) or len(decisions) != len(cases):
        raise Plan0064P4MeasurementError("P4 requires one decision for every review case.")
    expected_refs = [str(item.get("speaker_ref") or "") for item in cases]
    resolution_refs = [
        str(slot.get("speaker_ref") or "")
        for recording in resolution.get("recordings") or []
        for slot in recording.get("speaker_slots") or []
    ]
    allowed_people = {
        str(item.get("person_id") or "") for item in people if item.get("person_id")
    }
    if (
        len(expected_refs) != 39
        or len(set(expected_refs)) != 39
        or resolution_refs != expected_refs
        or len(allowed_people) != len(people)
    ):
        raise Plan0064P4MeasurementError("The P4 human-gold denominator drifted.")

    normalized: list[dict[str, Any]] = []
    for expected_ref, raw in zip(expected_refs, decisions, strict=True):
        if not isinstance(raw, Mapping) or set(raw) != {
            "speaker_ref",
            "decision",
            "person_id",
            "note",
        }:
            raise Plan0064P4MeasurementError("A P4 decision row has invalid fields.")
        speaker_ref = str(raw.get("speaker_ref") or "")
        decision = str(raw.get("decision") or "")
        person_id = raw.get("person_id")
        note = raw.get("note")
        if speaker_ref != expected_ref or decision not in DECISIONS:
            raise Plan0064P4MeasurementError("P4 decision order or value is invalid.")
        if not isinstance(note, str) or note != note.strip() or len(note) > 300:
            raise Plan0064P4MeasurementError("A P4 decision note is invalid.")
        if decision == "canonical_person":
            person_id = str(person_id or "")
            if person_id not in allowed_people:
                raise Plan0064P4MeasurementError(
                    "A P4 canonical-person decision escapes the reviewed option set."
                )
        elif person_id is not None:
            raise Plan0064P4MeasurementError(
                "Not-listed and unresolved decisions cannot carry a person ID."
            )
        normalized.append(
            {
                "speaker_ref": speaker_ref,
                "decision": decision,
                "person_id": person_id,
                "note": note,
            }
        )

    core = {
        "schema_version": GOLD_SCHEMA,
        "status": "complete_literal_human_gold",
        "authority_content_sha256": authority["content_sha256"],
        "p3_resolution_content_sha256": resolution["content_sha256"],
        "decision_count": len(normalized),
        "decision_type_counts": dict(
            sorted(Counter(item["decision"] for item in normalized).items())
        ),
        "decisions": normalized,
        "action_counts": dict(ACTION_COUNTS),
    }
    return _content_addressed(core)


def _candidate_lineage_complete(condition: str, slot: Mapping[str, Any]) -> bool:
    view = slot[condition]
    if view.get("disposition") != "candidate":
        return False
    acoustic = slot["acoustic"]
    context = slot["context"]
    acoustic_complete = bool(acoustic.get("probe_sha256")) and int(
        acoustic.get("supporting_model_count") or 0
    ) >= 2
    person_id = view.get("candidate_person_id")
    context_matches = [
        item
        for item in context.get("candidates") or []
        if isinstance(item, Mapping)
        and item.get("status") == "candidate_match"
        and item.get("prepared_person_id") == person_id
        and item.get("transcript_clue_ids")
        and item.get("provenance_source_ids")
    ]
    if condition == "acoustic":
        return acoustic_complete
    if condition == "context":
        return len(context_matches) == 1
    return acoustic_complete and len(context_matches) == 1


def _is_high_support(condition: str, slot: Mapping[str, Any]) -> bool:
    view = slot[condition]
    if view.get("disposition") != "candidate":
        return False
    if condition == "acoustic":
        return (
            view.get("confidence_band") == "high"
            and int(view.get("supporting_model_count") or 0) >= 2
        )
    if condition == "context":
        return _candidate_lineage_complete(condition, slot)
    return str(view.get("reason_code") or "") in HIGH_SUPPORT_REASONS


def _validated_development_gate(
    development_gate: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if development_gate is None:
        return {
            "present": False,
            "passed": False,
            "content_sha256": None,
            "reason": "reviewed_development_replay_missing",
        }
    _validate_content_addressed(development_gate, label="Development replay gate")
    if (
        development_gate.get("schema_version") != DEVELOPMENT_GATE_SCHEMA
        or development_gate.get("source_corpus") != "plan0063_reviewed_three_conversation"
        or development_gate.get("replay_exact") is not True
        or development_gate.get("high_support_wrong_count") != 0
        or development_gate.get("combined_correct_count", 0) < 1
        or development_gate.get("residual_correct_count", 0) < 1
        or any((development_gate.get("action_counts") or {}).values())
    ):
        return {
            "present": True,
            "passed": False,
            "content_sha256": development_gate.get("content_sha256"),
            "reason": "reviewed_development_replay_failed",
        }
    return {
        "present": True,
        "passed": True,
        "content_sha256": development_gate["content_sha256"],
        "reason": "reviewed_development_replay_passed",
    }


def recompute_measurement(
    *,
    authority: Mapping[str, Any],
    resolution: Mapping[str, Any],
    gold: Mapping[str, Any],
    development_gate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Score all four frozen conditions and evaluate a non-vacuous P5 gate."""

    _validate_content_addressed(authority, label="P4 review authority")
    _validate_content_addressed(resolution, label="P3 resolution")
    _validate_content_addressed(gold, label="P4 human gold")
    decisions = gold.get("decisions")
    if (
        gold.get("schema_version") != GOLD_SCHEMA
        or gold.get("authority_content_sha256") != authority.get("content_sha256")
        or gold.get("p3_resolution_content_sha256") != resolution.get("content_sha256")
        or gold.get("decision_count") != 39
        or not isinstance(decisions, list)
        or len(decisions) != 39
        or any((gold.get("action_counts") or {}).values())
    ):
        raise Plan0064P4MeasurementError("The frozen P4 human gold is invalid.")
    slot_rows = [
        slot
        for recording in resolution.get("recordings") or []
        for slot in recording.get("speaker_slots") or []
    ]
    if [item.get("speaker_ref") for item in decisions] != [
        item.get("speaker_ref") for item in slot_rows
    ]:
        raise Plan0064P4MeasurementError("P4 measurement and gold denominators differ.")

    counters = {condition: Counter() for condition in CONDITIONS}
    rows: list[dict[str, Any]] = []
    for decision, slot in zip(decisions, slot_rows, strict=True):
        gold_type = str(decision["decision"])
        gold_person = decision.get("person_id")
        condition_rows = []
        for condition in CONDITIONS:
            view = slot.get(condition)
            if not isinstance(view, Mapping):
                raise Plan0064P4MeasurementError("A P3 condition row is invalid.")
            disposition = str(view.get("disposition") or "")
            raw_person = view.get("candidate_person_id")
            if disposition == "candidate":
                proposed = str(raw_person or "")
                if not proposed:
                    raise Plan0064P4MeasurementError("A candidate has no person ID.")
            elif (
                condition == "acoustic"
                and disposition == "review"
                and view.get("reason_code") == "single_model_acoustic_support"
                and str(raw_person or "")
                and int(view.get("supporting_model_count") or 0) == 1
                and str(raw_person) in set(view.get("alternative_person_ids") or [])
            ):
                proposed = None
            elif raw_person is not None:
                raise Plan0064P4MeasurementError(
                    "A non-candidate P3 condition carries a person ID."
                )
            else:
                proposed = None
            if disposition not in {"candidate", "review", "abstain", "unavailable"}:
                raise Plan0064P4MeasurementError("A P3 disposition is invalid.")
            known = gold_type == "canonical_person"
            not_listed = gold_type == "not_listed"
            unresolved = gold_type == "unresolved"
            correct = bool(proposed and known and proposed == gold_person)
            wrong = bool(proposed and (not_listed or (known and proposed != gold_person)))
            unverifiable = bool(proposed and unresolved)
            high_support = _is_high_support(condition, slot)
            lineage_complete = (
                _candidate_lineage_complete(condition, slot) if proposed else False
            )
            counter = counters[condition]
            counter["evaluation_count"] += 1
            counter[f"{gold_type}_gold_count"] += 1
            counter[f"{disposition}_count"] += 1
            counter["correct_candidate_count"] += int(correct)
            counter["wrong_candidate_count"] += int(wrong)
            counter["unverifiable_candidate_count"] += int(unverifiable)
            counter["high_support_candidate_count"] += int(high_support)
            counter["high_support_wrong_count"] += int(high_support and wrong)
            counter["high_support_unverifiable_count"] += int(
                high_support and unverifiable
            )
            counter["candidate_lineage_complete_count"] += int(
                bool(proposed) and lineage_complete
            )
            residual_rule = (
                condition == "residual_policy"
                and view.get("reason_code")
                == "two_known_plus_one_independently_supported_residual"
            )
            counter["residual_rule_candidate_count"] += int(
                residual_rule and bool(proposed)
            )
            counter["residual_rule_correct_count"] += int(
                residual_rule and correct
            )
            counter["residual_rule_wrong_count"] += int(residual_rule and wrong)
            counter["residual_rule_lineage_complete_count"] += int(
                residual_rule and bool(proposed) and lineage_complete
            )
            condition_rows.append(
                {
                    "condition": condition,
                    "disposition": disposition,
                    "reason_code": view.get("reason_code"),
                    "proposed_person_id": proposed,
                    "correct_candidate": correct,
                    "wrong_candidate": wrong,
                    "unverifiable_candidate": unverifiable,
                    "high_support": high_support,
                    "candidate_lineage_complete": lineage_complete,
                }
            )
        rows.append(
            {
                "speaker_ref": decision["speaker_ref"],
                "gold_decision": gold_type,
                "gold_person_id": gold_person,
                "conditions": condition_rows,
            }
        )

    condition_metrics: dict[str, dict[str, Any]] = {}
    for condition, counter in counters.items():
        metrics = dict(counter)
        for key in (
            "canonical_person_gold_count",
            "not_listed_gold_count",
            "unresolved_gold_count",
            "candidate_count",
            "review_count",
            "abstain_count",
            "unavailable_count",
            "correct_candidate_count",
            "wrong_candidate_count",
            "unverifiable_candidate_count",
            "high_support_candidate_count",
            "high_support_wrong_count",
            "high_support_unverifiable_count",
            "candidate_lineage_complete_count",
            "residual_rule_candidate_count",
            "residual_rule_correct_count",
            "residual_rule_wrong_count",
            "residual_rule_lineage_complete_count",
        ):
            metrics.setdefault(key, 0)
        candidate_count = counter["candidate_count"]
        known_count = counter["canonical_person_gold_count"]
        metrics.update(
            {
                "candidate_precision": _ratio(
                    counter["correct_candidate_count"],
                    counter["correct_candidate_count"] + counter["wrong_candidate_count"],
                ),
                "known_person_recall": _ratio(
                    counter["correct_candidate_count"], known_count
                ),
                "abstention_rate": _ratio(counter["abstain_count"], 39),
                "review_rate": _ratio(counter["review_count"], 39),
                "unavailable_rate": _ratio(counter["unavailable_count"], 39),
                "candidate_lineage_completeness": _ratio(
                    counter["candidate_lineage_complete_count"], candidate_count
                ),
            }
        )
        condition_metrics[condition] = metrics

    development = _validated_development_gate(development_gate)
    combined = condition_metrics["combined"]
    residual = condition_metrics["residual_policy"]
    gate_checks = {
        "complete_39_row_human_gold": len(decisions) == 39,
        "source_disjoint_zero_high_support_wrong": sum(
            metrics["high_support_wrong_count"]
            for metrics in condition_metrics.values()
        )
        == 0,
        "source_disjoint_zero_high_support_unverifiable": sum(
            metrics["high_support_unverifiable_count"]
            for metrics in condition_metrics.values()
        )
        == 0,
        "combined_correct_acceptance_observed": combined["correct_candidate_count"] >= 1,
        "residual_correct_acceptance_observed": residual[
            "residual_rule_correct_count"
        ]
        >= 1,
        "combined_candidate_lineage_complete": (
            combined["candidate_count"] >= 1
            and combined["candidate_lineage_completeness"] == 1.0
        ),
        "residual_candidate_lineage_complete": (
            residual["residual_rule_candidate_count"] >= 1
            and residual["residual_rule_lineage_complete_count"]
            == residual["residual_rule_candidate_count"]
        ),
        "reviewed_development_replay_passed": development["passed"],
    }
    ready = all(gate_checks.values())
    core = {
        "schema_version": MEASUREMENT_SCHEMA,
        "status": "p4_measured_shadow_complete_no_apply",
        "authority_content_sha256": authority["content_sha256"],
        "p3_resolution_content_sha256": resolution["content_sha256"],
        "human_gold_content_sha256": gold["content_sha256"],
        "source_corpus": "chronological_source_disjoint_evaluation",
        "recording_count": len(resolution.get("recordings") or []),
        "speaker_slot_count": len(slot_rows),
        "condition_metrics": condition_metrics,
        "development_replay": development,
        "acceptance_gate": {
            "checks": gate_checks,
            "automatic_local_acceptance_ready": ready,
            "failed_checks": sorted(
                key for key, passed in gate_checks.items() if not passed
            ),
            "vacuous_zero_candidate_pass_forbidden": True,
        },
        "terminal_decision": "advance_to_p5" if ready else "withhold_p5",
        "apply_authorized": False,
        "rows": rows,
        "action_counts": dict(ACTION_COUNTS),
    }
    return _content_addressed(core)


def _paths(runtime_root: Path, gold_sha256: str) -> dict[str, Path]:
    root = runtime_root.expanduser().absolute()
    run = root / f"p4-measurement-{gold_sha256[:24]}"
    return {
        "root": root,
        "run": run,
        "gold": run / "human-gold.json",
        "gold_receipt": run / "human-gold-receipt.json",
        "measurement": run / "measurement.json",
        "measurement_receipt": run / "measurement-receipt.json",
        "terminal": run / "terminal.json",
    }


def freeze_human_gold_and_measurement(
    submission: Mapping[str, Any],
    *,
    p0_content_sha256: str,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    development_gate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Freeze exact P4 gold and comparison while preserving the no-apply seam."""

    authority, _p3_receipt, resolution = _authorities(
        p0_content_sha256, runtime_root=runtime_root
    )
    gold = normalize_human_gold(
        submission, authority=authority, resolution=resolution
    )
    measurement = recompute_measurement(
        authority=authority,
        resolution=resolution,
        gold=gold,
        development_gate=development_gate,
    )
    paths = _paths(runtime_root, gold["content_sha256"])
    if paths["terminal"].exists():
        return replay_human_gold_and_measurement(
            gold_content_sha256=gold["content_sha256"],
            p0_content_sha256=p0_content_sha256,
            runtime_root=runtime_root,
            development_gate=development_gate,
        )
    if paths["run"].exists():
        raise Plan0064P4MeasurementError("A partial P4 measurement directory exists.")
    ensure_private_tree(paths["root"], paths["run"])
    write_immutable_private_json(paths["gold"], gold)
    gold_receipt = _content_addressed(
        {
            "schema_version": GOLD_RECEIPT_SCHEMA,
            "status": "human_gold_frozen",
            "human_gold_content_sha256": gold["content_sha256"],
            "human_gold_file_sha256": sha256_file(paths["gold"]),
            "decision_count": 39,
            "action_counts": dict(ACTION_COUNTS),
        }
    )
    write_immutable_private_json(paths["gold_receipt"], gold_receipt)
    write_immutable_private_json(paths["measurement"], measurement)
    measurement_receipt = _content_addressed(
        {
            "schema_version": MEASUREMENT_RECEIPT_SCHEMA,
            "status": "measurement_frozen_no_apply",
            "human_gold_content_sha256": gold["content_sha256"],
            "measurement_content_sha256": measurement["content_sha256"],
            "measurement_file_sha256": sha256_file(paths["measurement"]),
            "terminal_decision": measurement["terminal_decision"],
            "action_counts": dict(ACTION_COUNTS),
        }
    )
    write_immutable_private_json(paths["measurement_receipt"], measurement_receipt)
    terminal = _content_addressed(
        {
            "schema_version": TERMINAL_SCHEMA,
            "status": "complete_no_apply",
            "human_gold_content_sha256": gold["content_sha256"],
            "measurement_content_sha256": measurement["content_sha256"],
            "human_gold_file_sha256": sha256_file(paths["gold"]),
            "measurement_file_sha256": sha256_file(paths["measurement"]),
            "terminal_decision": measurement["terminal_decision"],
            "failed_checks": measurement["acceptance_gate"]["failed_checks"],
            "metrics_recomputed": recompute_measurement(
                authority=authority,
                resolution=resolution,
                gold=gold,
                development_gate=development_gate,
            )
            == measurement,
            "apply_authorized": False,
            "action_counts": dict(ACTION_COUNTS),
        }
    )
    if terminal["metrics_recomputed"] is not True:
        raise Plan0064P4MeasurementError("Independent P4 recomputation disagreed.")
    write_immutable_private_json(paths["terminal"], terminal)
    return {
        **terminal,
        "private_terminal_path": str(paths["terminal"]),
        "private_measurement_path": str(paths["measurement"]),
        "idempotent_replay": False,
    }


def replay_human_gold_and_measurement(
    *,
    gold_content_sha256: str,
    p0_content_sha256: str,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    development_gate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Recompute and verify a frozen P4 measurement against current authorities."""

    authority, _p3_receipt, resolution = _authorities(
        p0_content_sha256, runtime_root=runtime_root
    )
    paths = _paths(runtime_root, gold_content_sha256)
    for key in ("gold", "gold_receipt", "measurement", "measurement_receipt", "terminal"):
        require_private_file(paths[key], paths["root"])
    gold = _read(paths["gold"])
    gold_receipt = _read(paths["gold_receipt"])
    measurement = _read(paths["measurement"])
    measurement_receipt = _read(paths["measurement_receipt"])
    terminal = _read(paths["terminal"])
    expected = recompute_measurement(
        authority=authority,
        resolution=resolution,
        gold=gold,
        development_gate=development_gate,
    )
    for value, label in (
        (gold_receipt, "Human-gold receipt"),
        (measurement_receipt, "Measurement receipt"),
        (terminal, "P4 terminal"),
    ):
        _validate_content_addressed(value, label=label)
    if (
        gold.get("content_sha256") != gold_content_sha256
        or measurement != expected
        or gold_receipt.get("schema_version") != GOLD_RECEIPT_SCHEMA
        or gold_receipt.get("human_gold_file_sha256") != sha256_file(paths["gold"])
        or measurement_receipt.get("schema_version") != MEASUREMENT_RECEIPT_SCHEMA
        or measurement_receipt.get("measurement_file_sha256")
        != sha256_file(paths["measurement"])
        or terminal.get("schema_version") != TERMINAL_SCHEMA
        or terminal.get("human_gold_file_sha256") != sha256_file(paths["gold"])
        or terminal.get("measurement_file_sha256") != sha256_file(paths["measurement"])
        or terminal.get("terminal_decision") != measurement["terminal_decision"]
        or terminal.get("failed_checks")
        != measurement["acceptance_gate"]["failed_checks"]
        or terminal.get("metrics_recomputed") is not True
        or terminal.get("apply_authorized") is not False
        or any((terminal.get("action_counts") or {}).values())
    ):
        raise Plan0064P4MeasurementError("The frozen P4 measurement drifted.")
    return {
        **terminal,
        "private_terminal_path": str(paths["terminal"]),
        "private_measurement_path": str(paths["measurement"]),
        "idempotent_replay": True,
    }


def _load_optional(path: Path | None) -> dict[str, Any] | None:
    return _read(path.expanduser().absolute()) if path is not None else None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "replay"))
    parser.add_argument("--p0-content-sha256", required=True)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--decision-file", type=Path)
    parser.add_argument("--gold-content-sha256")
    parser.add_argument("--development-gate", type=Path)
    args = parser.parse_args(argv)
    development_gate = _load_optional(args.development_gate)
    if args.action == "freeze":
        if args.decision_file is None or args.gold_content_sha256:
            parser.error("freeze requires --decision-file and forbids --gold-content-sha256")
        result = freeze_human_gold_and_measurement(
            _read(args.decision_file.expanduser().absolute()),
            p0_content_sha256=args.p0_content_sha256,
            runtime_root=args.runtime_root,
            development_gate=development_gate,
        )
    else:
        if not args.gold_content_sha256 or args.decision_file is not None:
            parser.error("replay requires --gold-content-sha256 and forbids --decision-file")
        result = replay_human_gold_and_measurement(
            gold_content_sha256=args.gold_content_sha256,
            p0_content_sha256=args.p0_content_sha256,
            runtime_root=args.runtime_root,
            development_gate=development_gate,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
