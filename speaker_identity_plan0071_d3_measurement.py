"""Freeze and measure Plan 0071 D3 literal supplemental human gold."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import speaker_identity_plan0071_d2_predictions as predictions
import speaker_identity_plan0071_d2_predictions_attempt2 as attempt2
import speaker_identity_plan0071_d2_review as review
from acoustic_audio_derivatives import (
    canonical_artifact_hash,
    ensure_private_tree,
    read_private_object,
    require_private_file,
    sha256_file,
    write_immutable_private_json,
)


DECISION_SCHEMA = review.DECISION_SCHEMA
REVIEW_AUTHORITY_SCHEMA = review.AUTHORITY_SCHEMA
PREDICTION_RESOLUTION_SCHEMA = predictions.RESOLUTION_SCHEMA
GOLD_SCHEMA = "transcribe-audio.plan0071-d3-human-gold.v1"
GOLD_RECEIPT_SCHEMA = "transcribe-audio.plan0071-d3-human-gold-receipt.v1"
MEASUREMENT_SCHEMA = "transcribe-audio.plan0071-d3-measurement.v1"
MEASUREMENT_RECEIPT_SCHEMA = "transcribe-audio.plan0071-d3-measurement-receipt.v1"
TERMINAL_SCHEMA = "transcribe-audio.plan0071-d3-terminal.v1"
DECISIONS = ("canonical_person", "not_listed", "unresolved")
CONDITIONS = ("acoustic", "context", "combined", "residual_policy")
MUTATION_EFFECT_COUNTS = dict(review.MUTATION_EFFECT_COUNTS)
DEFAULT_RUNTIME_ROOT = review.DEFAULT_RUNTIME_ROOT
REVIEW_PREVIEW_CONTENT_SHA256 = (
    "315abbab69590337bb0adcd7457b56bd8812a532cb2bc1725b78b8eaa9e482e9"
)
REVIEW_AUTHORITY_CONTENT_SHA256 = (
    "3aac595b1dce3ba6b8e41d2de653fb399b56cd21841e782e505bbf2cf34c91ba"
)
REVIEW_RECEIPT_CONTENT_SHA256 = (
    "996e92c9abe5e4c394b0f7291d32901b1bb68d7c943e07447b4ec9a645d9cabf"
)
PREDICTION_RECEIPT_CONTENT_SHA256 = review.PREDICTION_RECEIPT_CONTENT_SHA256
PREDICTION_RESOLUTION_CONTENT_SHA256 = review.PREDICTION_RESOLUTION_CONTENT_SHA256


class Plan0071D3MeasurementError(ValueError):
    """Raised when D3 human gold, measurement, or lineage is not exact."""


def _content(value: Mapping[str, Any]) -> dict[str, Any]:
    core = dict(value)
    core.pop("content_sha256", None)
    return {**core, "content_sha256": canonical_artifact_hash(core)}


def _validate_content(value: Mapping[str, Any], label: str) -> None:
    core = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != canonical_artifact_hash(core):
        raise Plan0071D3MeasurementError(f"{label} content hash drifted.")


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


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


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], check=False, capture_output=True, text=True
    )
    if result.returncode:
        raise Plan0071D3MeasurementError(
            result.stderr.strip() or "Git authority read failed."
        )
    return result.stdout.strip()


def _source_authority(*, require_clean: bool) -> dict[str, Any]:
    module = Path(__file__).resolve()
    root = Path(_git("rev-parse", "--show-toplevel")).resolve()
    relative = module.relative_to(root).as_posix()
    commit = _git("log", "-1", "--format=%H", "--", relative)
    committed = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        check=False,
        capture_output=True,
    )
    upstream = _git("rev-parse", "@{upstream}")
    module_sha256 = hashlib.sha256(module.read_bytes()).hexdigest()
    value = {
        "module_name": relative,
        "module_commit": commit,
        "module_sha256": module_sha256,
        "module_blob_matches": (
            committed.returncode == 0
            and module_sha256 == hashlib.sha256(committed.stdout).hexdigest()
        ),
        "clean": not _git("status", "--porcelain=v1"),
        "upstream_ahead": int(_git("rev-list", "--count", f"{upstream}..HEAD")),
        "upstream_behind": int(_git("rev-list", "--count", f"HEAD..{upstream}")),
    }
    if value["module_blob_matches"] is not True or (
        require_clean
        and (
            value["clean"] is not True
            or value["upstream_ahead"]
            or value["upstream_behind"]
        )
    ):
        raise Plan0071D3MeasurementError(
            "D3 measurement source is not committed, clean, and upstream-even."
        )
    return value


def _authorities(runtime_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    prediction_replay = attempt2.replay_attempt2(runtime_root=runtime_root)
    review_replay = review.replay_review(runtime_root=runtime_root)
    review_paths = review._paths(runtime_root, REVIEW_PREVIEW_CONTENT_SHA256)
    prediction_paths = attempt2._paths(runtime_root)
    for path, root in (
        (review_paths["authority"], review_paths["run"]),
        (prediction_paths["resolution"], prediction_paths["run"]),
    ):
        require_private_file(path, root)
    authority = read_private_object(review_paths["authority"])
    resolution = read_private_object(prediction_paths["resolution"])
    _validate_content(authority, "D2 review authority")
    _validate_content(resolution, "D2 prediction resolution")
    authority_refs = [
        str(item.get("speaker_ref") or "") for item in authority.get("cases") or []
    ]
    resolution_refs = [
        str(slot.get("speaker_ref") or "")
        for recording in resolution.get("recordings") or []
        for slot in recording.get("speaker_slots") or []
    ]
    if (
        prediction_replay.get("content_sha256")
        != PREDICTION_RECEIPT_CONTENT_SHA256
        or review_replay.get("content_sha256") != REVIEW_RECEIPT_CONTENT_SHA256
        or authority.get("content_sha256") != REVIEW_AUTHORITY_CONTENT_SHA256
        or resolution.get("content_sha256")
        != PREDICTION_RESOLUTION_CONTENT_SHA256
        or authority_refs != resolution_refs
        or len(authority_refs) != 18
        or len(set(authority_refs)) != 18
        or authority.get("human_decision_count") != 0
        or authority.get("model_predictions_visible") is not False
        or resolution.get("contains_gold") is not False
        or authority.get("mutation_effect_counts") != MUTATION_EFFECT_COUNTS
        or resolution.get("mutation_effect_counts") != MUTATION_EFFECT_COUNTS
    ):
        raise Plan0071D3MeasurementError("The frozen D2 authority drifted.")
    return authority, resolution


def _paths(runtime_root: Path) -> dict[str, Path]:
    root = runtime_root.expanduser().resolve()
    run = root / f"d3-measurement-{REVIEW_AUTHORITY_CONTENT_SHA256[:24]}"
    return {
        "root": root,
        "run": run,
        "gold": run / "human-gold.json",
        "gold_receipt": run / "human-gold-receipt.json",
        "measurement": run / "measurement.json",
        "measurement_receipt": run / "measurement-receipt.json",
        "terminal": run / "terminal.json",
    }


def normalize_human_gold(
    submission: Mapping[str, Any],
    *,
    authority: Mapping[str, Any],
    resolution: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one exact complete client export and freeze its literal meaning."""

    _validate_content(authority, "D2 review authority")
    _validate_content(resolution, "D2 prediction resolution")
    if set(submission) != {"schema_version", "authority_content_sha256", "decisions"}:
        raise Plan0071D3MeasurementError(
            "The D2 decision export has unknown or missing fields."
        )
    if (
        submission.get("schema_version") != DECISION_SCHEMA
        or submission.get("authority_content_sha256")
        != authority.get("content_sha256")
        or authority.get("schema_version") != REVIEW_AUTHORITY_SCHEMA
        or resolution.get("schema_version") != PREDICTION_RESOLUTION_SCHEMA
        or authority.get("human_decision_count") != 0
        or authority.get("model_predictions_visible") is not False
        or authority.get("mutation_effect_counts") != MUTATION_EFFECT_COUNTS
        or resolution.get("contains_gold") is not False
        or resolution.get("mutation_effect_counts") != MUTATION_EFFECT_COUNTS
    ):
        raise Plan0071D3MeasurementError(
            "The D2 decision export or its frozen authority drifted."
        )
    cases = authority.get("cases")
    people = authority.get("people")
    decisions = submission.get("decisions")
    if not isinstance(cases, list) or not isinstance(people, list):
        raise Plan0071D3MeasurementError("The D2 review authority is malformed.")
    if not isinstance(decisions, list) or len(decisions) != len(cases):
        raise Plan0071D3MeasurementError(
            "D3 requires one decision for every review case."
        )
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
        len(expected_refs) != 18
        or len(set(expected_refs)) != 18
        or resolution_refs != expected_refs
        or len(allowed_people) != len(people)
    ):
        raise Plan0071D3MeasurementError("The D3 human-gold denominator drifted.")

    normalized: list[dict[str, Any]] = []
    for expected_ref, raw in zip(expected_refs, decisions, strict=True):
        if not isinstance(raw, Mapping) or set(raw) != {
            "speaker_ref",
            "decision",
            "person_id",
            "note",
        }:
            raise Plan0071D3MeasurementError("A D3 decision row has invalid fields.")
        speaker_ref = str(raw.get("speaker_ref") or "")
        decision = str(raw.get("decision") or "")
        person_id = raw.get("person_id")
        note = raw.get("note")
        if speaker_ref != expected_ref or decision not in DECISIONS:
            raise Plan0071D3MeasurementError("D3 decision order or value is invalid.")
        if not isinstance(note, str) or note != note.strip() or len(note) > 300:
            raise Plan0071D3MeasurementError("A D3 decision note is invalid.")
        if decision == "canonical_person":
            person_id = str(person_id or "")
            if person_id not in allowed_people:
                raise Plan0071D3MeasurementError(
                    "A D3 canonical-person decision escapes the reviewed option set."
                )
        elif person_id is not None:
            raise Plan0071D3MeasurementError(
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

    return _content(
        {
            "schema_version": GOLD_SCHEMA,
            "status": "complete_literal_human_gold",
            "authority_content_sha256": authority["content_sha256"],
            "prediction_resolution_content_sha256": resolution["content_sha256"],
            "decision_count": len(normalized),
            "decision_type_counts": dict(
                sorted(Counter(item["decision"] for item in normalized).items())
            ),
            "decisions": normalized,
            "mutation_effect_counts": dict(MUTATION_EFFECT_COUNTS),
        }
    )


def measure_d3(
    *,
    authority: Mapping[str, Any],
    resolution: Mapping[str, Any],
    gold: Mapping[str, Any],
) -> dict[str, Any]:
    """Score the frozen D2 predictions once and decide the D3 terminal."""

    _validate_content(authority, "D2 review authority")
    _validate_content(resolution, "D2 prediction resolution")
    _validate_content(gold, "D3 human gold")
    decisions = gold.get("decisions")
    slots = [
        slot
        for recording in resolution.get("recordings") or []
        for slot in recording.get("speaker_slots") or []
    ]
    if (
        gold.get("schema_version") != GOLD_SCHEMA
        or gold.get("authority_content_sha256") != authority.get("content_sha256")
        or gold.get("prediction_resolution_content_sha256")
        != resolution.get("content_sha256")
        or gold.get("decision_count") != 18
        or not isinstance(decisions, list)
        or len(decisions) != 18
        or len(slots) != 18
        or gold.get("mutation_effect_counts") != MUTATION_EFFECT_COUNTS
        or [item.get("speaker_ref") for item in decisions]
        != [item.get("speaker_ref") for item in slots]
    ):
        raise Plan0071D3MeasurementError(
            "The frozen D3 human gold or prediction denominator is invalid."
        )

    counters = {condition: Counter() for condition in CONDITIONS}
    rows: list[dict[str, Any]] = []
    for decision, slot in zip(decisions, slots, strict=True):
        gold_type = str(decision["decision"])
        gold_person = decision.get("person_id")
        condition_rows = []
        for condition in CONDITIONS:
            view = slot.get(condition)
            if not isinstance(view, Mapping):
                raise Plan0071D3MeasurementError(
                    "A D2 prediction condition row is invalid."
                )
            disposition = str(view.get("disposition") or "")
            raw_person = view.get("candidate_person_id")
            if disposition == "candidate":
                proposed = str(raw_person or "")
                if not proposed:
                    raise Plan0071D3MeasurementError(
                        "A D2 prediction candidate has no person ID."
                    )
            elif raw_person is not None:
                raise Plan0071D3MeasurementError(
                    "A non-candidate D2 prediction carries a person ID."
                )
            else:
                proposed = None
            if disposition not in {"candidate", "review", "abstain", "unavailable"}:
                raise Plan0071D3MeasurementError(
                    "A D2 prediction disposition is invalid."
                )
            known = gold_type == "canonical_person"
            not_listed = gold_type == "not_listed"
            unresolved = gold_type == "unresolved"
            correct = bool(proposed and known and proposed == gold_person)
            wrong = bool(proposed and (not_listed or (known and proposed != gold_person)))
            unverifiable = bool(proposed and unresolved)
            lineage_complete = (
                _candidate_lineage_complete(condition, slot) if proposed else False
            )
            reason_code = str(view.get("reason_code") or "")
            pillar_agreement = reason_code in {
                "pillar_agreement",
                "pillar_agreement_same_person_multi_label",
            }
            residual_rule = (
                condition == "residual_policy"
                and reason_code
                == "two_known_plus_one_independently_supported_residual"
            )
            counter = counters[condition]
            counter["evaluation_count"] += 1
            counter[f"{gold_type}_gold_count"] += 1
            counter[f"{disposition}_count"] += 1
            counter["correct_candidate_count"] += int(correct)
            counter["wrong_candidate_count"] += int(wrong)
            counter["unverifiable_candidate_count"] += int(unverifiable)
            counter["candidate_lineage_complete_count"] += int(
                bool(proposed) and lineage_complete
            )
            counter["pillar_agreement_candidate_count"] += int(
                pillar_agreement and bool(proposed)
            )
            counter["pillar_agreement_correct_count"] += int(
                pillar_agreement and correct
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
                    "reason_code": reason_code,
                    "proposed_person_id": proposed,
                    "correct_candidate": correct,
                    "wrong_candidate": wrong,
                    "unverifiable_candidate": unverifiable,
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

    metric_keys = (
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
        "candidate_lineage_complete_count",
        "pillar_agreement_candidate_count",
        "pillar_agreement_correct_count",
        "residual_rule_candidate_count",
        "residual_rule_correct_count",
        "residual_rule_wrong_count",
        "residual_rule_lineage_complete_count",
    )
    condition_metrics: dict[str, dict[str, Any]] = {}
    for condition, counter in counters.items():
        metrics = dict(counter)
        for key in metric_keys:
            metrics.setdefault(key, 0)
        candidate_count = metrics["candidate_count"]
        metrics.update(
            {
                "candidate_precision": _ratio(
                    metrics["correct_candidate_count"],
                    metrics["correct_candidate_count"]
                    + metrics["wrong_candidate_count"],
                ),
                "known_person_recall": _ratio(
                    metrics["correct_candidate_count"],
                    metrics["canonical_person_gold_count"],
                ),
                "candidate_lineage_completeness": _ratio(
                    metrics["candidate_lineage_complete_count"], candidate_count
                ),
            }
        )
        condition_metrics[condition] = metrics

    combined = condition_metrics["combined"]
    residual = condition_metrics["residual_policy"]
    joined_residual_candidate_count = (
        combined["candidate_count"] + residual["candidate_count"]
    )
    joined_residual_lineage_count = (
        combined["candidate_lineage_complete_count"]
        + residual["candidate_lineage_complete_count"]
    )
    gate_checks = {
        "complete_18_row_human_gold": len(decisions) == 18,
        "pillar_agreement_correct_acceptance_observed": combined[
            "pillar_agreement_correct_count"
        ]
        >= 1,
        "residual_correct_acceptance_observed": residual[
            "residual_rule_correct_count"
        ]
        >= 1,
        "joined_residual_zero_wrong_identities": (
            combined["wrong_candidate_count"] + residual["wrong_candidate_count"]
        )
        == 0,
        "joined_residual_candidate_lineage_complete": (
            joined_residual_lineage_count == joined_residual_candidate_count
        ),
        "prediction_remained_gold_blind": resolution.get("contains_gold") is False,
    }
    ready = all(gate_checks.values())
    if ready:
        terminal_decision = "advance_to_fresh_evaluation"
    elif residual["residual_rule_candidate_count"] == 0:
        terminal_decision = "residual_population_infeasible"
    else:
        terminal_decision = "withhold_fresh_evaluation"
    return _content(
        {
            "schema_version": MEASUREMENT_SCHEMA,
            "status": "d3_measured_shadow_complete_no_apply",
            "authority_content_sha256": authority["content_sha256"],
            "prediction_resolution_content_sha256": resolution["content_sha256"],
            "human_gold_content_sha256": gold["content_sha256"],
            "recording_count": len(resolution.get("recordings") or []),
            "speaker_slot_count": len(slots),
            "condition_metrics": condition_metrics,
            "acceptance_gate": {
                "checks": gate_checks,
                "passed": ready,
                "failed_checks": sorted(
                    key for key, passed in gate_checks.items() if not passed
                ),
                "vacuous_zero_candidate_pass_forbidden": True,
            },
            "terminal_decision": terminal_decision,
            "fresh_evaluation_allowed": ready,
            "apply_authorized": False,
            "rows": rows,
            "mutation_effect_counts": dict(MUTATION_EFFECT_COUNTS),
        }
    )


def replay_d3(*, runtime_root: Path = DEFAULT_RUNTIME_ROOT) -> dict[str, Any]:
    """Recompute and verify the one frozen D3 measurement."""

    authority, resolution = _authorities(runtime_root)
    paths = _paths(runtime_root)
    for key in (
        "gold",
        "gold_receipt",
        "measurement",
        "measurement_receipt",
        "terminal",
    ):
        require_private_file(paths[key], paths["run"])
    gold = read_private_object(paths["gold"])
    gold_receipt = read_private_object(paths["gold_receipt"])
    measurement = read_private_object(paths["measurement"])
    measurement_receipt = read_private_object(paths["measurement_receipt"])
    terminal = read_private_object(paths["terminal"])
    expected_measurement = measure_d3(
        authority=authority,
        resolution=resolution,
        gold=gold,
    )
    for value, label in (
        (gold, "D3 human gold"),
        (gold_receipt, "D3 human-gold receipt"),
        (measurement, "D3 measurement"),
        (measurement_receipt, "D3 measurement receipt"),
        (terminal, "D3 terminal"),
    ):
        _validate_content(value, label)
    current_source = _source_authority(require_clean=False)
    if (
        measurement != expected_measurement
        or gold_receipt.get("schema_version") != GOLD_RECEIPT_SCHEMA
        or gold_receipt.get("human_gold_content_sha256")
        != gold.get("content_sha256")
        or gold_receipt.get("human_gold_file_sha256") != sha256_file(paths["gold"])
        or measurement_receipt.get("schema_version")
        != MEASUREMENT_RECEIPT_SCHEMA
        or measurement_receipt.get("measurement_content_sha256")
        != measurement.get("content_sha256")
        or measurement_receipt.get("measurement_file_sha256")
        != sha256_file(paths["measurement"])
        or terminal.get("schema_version") != TERMINAL_SCHEMA
        or terminal.get("human_gold_content_sha256") != gold.get("content_sha256")
        or terminal.get("human_gold_file_sha256") != sha256_file(paths["gold"])
        or terminal.get("measurement_content_sha256")
        != measurement.get("content_sha256")
        or terminal.get("measurement_file_sha256")
        != sha256_file(paths["measurement"])
        or terminal.get("terminal_decision") != measurement.get("terminal_decision")
        or terminal.get("failed_checks")
        != measurement.get("acceptance_gate", {}).get("failed_checks")
        or terminal.get("fresh_evaluation_allowed")
        is not measurement.get("fresh_evaluation_allowed")
        or terminal.get("apply_authorized") is not False
        or terminal.get("mutation_effect_counts") != MUTATION_EFFECT_COUNTS
        or terminal.get("source_authority", {}).get("module_sha256")
        != current_source.get("module_sha256")
    ):
        raise Plan0071D3MeasurementError("The frozen D3 measurement drifted.")
    return {
        **terminal,
        "private_run_root": str(paths["run"]),
        "private_human_gold_path": str(paths["gold"]),
        "private_measurement_path": str(paths["measurement"]),
        "private_terminal_path": str(paths["terminal"]),
        "idempotent_replay": True,
    }


def freeze_d3(
    submission: Mapping[str, Any],
    *,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
) -> dict[str, Any]:
    """Freeze one exact D3 gold submission and its deterministic terminal."""

    authority, resolution = _authorities(runtime_root)
    gold = normalize_human_gold(
        submission,
        authority=authority,
        resolution=resolution,
    )
    paths = _paths(runtime_root)
    if paths["terminal"].exists():
        replay = replay_d3(runtime_root=runtime_root)
        stored = read_private_object(paths["gold"])
        if stored.get("content_sha256") != gold.get("content_sha256"):
            raise Plan0071D3MeasurementError(
                "D3 was already measured from a different literal gold submission."
            )
        return replay

    source_authority = _source_authority(require_clean=True)
    measurement = measure_d3(
        authority=authority,
        resolution=resolution,
        gold=gold,
    )
    ensure_private_tree(paths["root"], paths["run"])
    write_immutable_private_json(paths["gold"], gold)
    gold_receipt = _content(
        {
            "schema_version": GOLD_RECEIPT_SCHEMA,
            "status": "literal_human_gold_frozen",
            "authority_content_sha256": authority["content_sha256"],
            "prediction_resolution_content_sha256": resolution["content_sha256"],
            "submission_sha256": canonical_artifact_hash(dict(submission)),
            "human_gold_content_sha256": gold["content_sha256"],
            "human_gold_file_sha256": sha256_file(paths["gold"]),
            "decision_count": gold["decision_count"],
            "decision_type_counts": gold["decision_type_counts"],
            "mutation_effect_counts": dict(MUTATION_EFFECT_COUNTS),
        }
    )
    write_immutable_private_json(paths["gold_receipt"], gold_receipt)
    write_immutable_private_json(paths["measurement"], measurement)
    measurement_receipt = _content(
        {
            "schema_version": MEASUREMENT_RECEIPT_SCHEMA,
            "status": "d3_measurement_frozen",
            "human_gold_content_sha256": gold["content_sha256"],
            "measurement_content_sha256": measurement["content_sha256"],
            "measurement_file_sha256": sha256_file(paths["measurement"]),
            "terminal_decision": measurement["terminal_decision"],
            "fresh_evaluation_allowed": measurement["fresh_evaluation_allowed"],
            "mutation_effect_counts": dict(MUTATION_EFFECT_COUNTS),
        }
    )
    write_immutable_private_json(paths["measurement_receipt"], measurement_receipt)
    decision = measurement["terminal_decision"]
    terminal = _content(
        {
            "schema_version": TERMINAL_SCHEMA,
            "status": (
                "d3_passed_fresh_evaluation_gate"
                if decision == "advance_to_fresh_evaluation"
                else "d3_closed_residual_population_infeasible"
                if decision == "residual_population_infeasible"
                else "d3_withheld_fresh_evaluation"
            ),
            "source_authority": source_authority,
            "review_authority_content_sha256": authority["content_sha256"],
            "prediction_receipt_content_sha256": PREDICTION_RECEIPT_CONTENT_SHA256,
            "prediction_resolution_content_sha256": resolution["content_sha256"],
            "human_gold_content_sha256": gold["content_sha256"],
            "human_gold_file_sha256": sha256_file(paths["gold"]),
            "measurement_content_sha256": measurement["content_sha256"],
            "measurement_file_sha256": sha256_file(paths["measurement"]),
            "terminal_decision": decision,
            "failed_checks": measurement["acceptance_gate"]["failed_checks"],
            "condition_metrics": measurement["condition_metrics"],
            "prediction_residual_acceptance_count": resolution.get("summary", {}).get(
                "residual_acceptance_count"
            ),
            "human_decision_count": gold["decision_count"],
            "fresh_evaluation_allowed": measurement["fresh_evaluation_allowed"],
            "next_packet": "E0" if measurement["fresh_evaluation_allowed"] else None,
            "apply_authorized": False,
            "mutation_effect_counts": dict(MUTATION_EFFECT_COUNTS),
        }
    )
    write_immutable_private_json(paths["terminal"], terminal)
    return {
        **terminal,
        "private_run_root": str(paths["run"]),
        "private_human_gold_path": str(paths["gold"]),
        "private_measurement_path": str(paths["measurement"]),
        "private_terminal_path": str(paths["terminal"]),
        "idempotent_replay": False,
    }


def _submission(path: str) -> dict[str, Any]:
    if path == "-":
        value = json.load(sys.stdin)
    else:
        value = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Plan0071D3MeasurementError("The D3 submission must be a JSON object.")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--submission", metavar="PATH", help="Use - to read JSON on stdin.")
    group.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    result = (
        replay_d3()
        if args.replay
        else freeze_d3(_submission(str(args.submission)))
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
