#!/usr/bin/env python3
"""Plan 0065 D1 acoustic safety recovery and development gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from array import array
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import acoustic_verification as verification
from acoustic_audio_derivatives import (
    canonical_artifact_hash,
    ensure_private_tree,
    read_private_object,
    require_private_file,
    sha256_file,
    write_immutable_private_json,
)
import speaker_identity_plan0064_p1 as p1
import speaker_identity_plan0065_d0 as d0


DEFAULT_RUNTIME_ROOT = d0.DEFAULT_RUNTIME_ROOT
DEFAULT_D0_MANIFEST = (
    DEFAULT_RUNTIME_ROOT.expanduser()
    / "d0-a38891bbf46decbf74ba04b9"
    / "private-manifest.json"
)
DEFAULT_D0_RECEIPT = DEFAULT_D0_MANIFEST.with_name("receipt.json")
DEFAULT_CALIBRATION_ROOT = Path(
    "~/.local/state/transcribe-audio/plan-0037/generation-3/"
    "recalibration-executions/"
    "generation3-recalibration-execution-39298c74aab4a773945268cd"
)
DEFAULT_THRESHOLD_APPLICATION = DEFAULT_CALIBRATION_ROOT / "threshold-application.json"
DEFAULT_SCORE_MATRIX = DEFAULT_CALIBRATION_ROOT / "score-matrix.json"

D0_MANIFEST_SHA256 = "a38891bbf46decbf74ba04b9103eab1c435a5d33b000111ab06f876a7907c6c4"
D0_RECEIPT_SHA256 = "14290463a4c7f8f3701c3aa21c08cc898139e0a2a7a74e038c7946cdd6a998f6"
CALIBRATION_SCORE_MATRIX_FILE_SHA256 = (
    "3fb983b06b1984724c2f0e3e3c01f55065ff755e36416260c33fe0f2649201c2"
)
CALIBRATION_THRESHOLD_FILE_SHA256 = (
    "308f326d3fe9baa175ed32c90df4255a8d4bfc1924c6f925eab490ae2832f4d1"
)

POLICY_SCHEMA = "transcribe-audio.plan0065-d1-acoustic-safety-policy.v1"
EVIDENCE_SCHEMA = "transcribe-audio.plan0065-d1-development-evidence.v1"
RECEIPT_SCHEMA = "transcribe-audio.plan0065-d1-receipt.v1"
SAFETY_NUMERATOR = 2
SAFETY_DENOMINATOR = 3

ACTION_COUNTS = dict(d0.ACTION_COUNTS)


class Plan0065D1Error(ValueError):
    """Raised when D1 authority, evidence, or safety invariants fail."""


def _hash(value: Any) -> str:
    return canonical_artifact_hash(value)


def _content_addressed(value: Mapping[str, Any]) -> dict[str, Any]:
    core = dict(value)
    core.pop("content_sha256", None)
    return {**core, "content_sha256": _hash(core)}


def _validate_content(value: Mapping[str, Any], *, label: str) -> None:
    core = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != _hash(core):
        raise Plan0065D1Error(f"{label} content hash drifted.")


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Plan0065D1Error(f"Expected an object at {path}.")
    return value


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode:
        raise Plan0065D1Error(result.stderr.strip() or "Git authority read failed.")
    return result.stdout.strip()


def repository_authority() -> dict[str, Any]:
    module = Path(__file__).resolve()
    root = Path(_git("rev-parse", "--show-toplevel")).resolve()
    relative = module.relative_to(root).as_posix()
    module_commit = _git("log", "-1", "--format=%H", "--", relative)
    if not module_commit:
        raise Plan0065D1Error("The D1 module has no committed authority.")
    committed = subprocess.run(
        ["git", "show", f"{module_commit}:{relative}"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if committed.returncode:
        raise Plan0065D1Error("The committed D1 module cannot be read.")
    upstream = _git("rev-parse", "@{upstream}")
    authority = {
        "module_name": relative,
        "module_commit": module_commit,
        "module_sha256": hashlib.sha256(module.read_bytes()).hexdigest(),
        "module_blob_matches": (
            hashlib.sha256(module.read_bytes()).hexdigest()
            == hashlib.sha256(committed.stdout).hexdigest()
        ),
        "clean": not _git("status", "--porcelain=v1"),
        "upstream_ahead": int(_git("rev-list", "--count", f"{upstream}..HEAD")),
        "upstream_behind": int(_git("rev-list", "--count", f"HEAD..{upstream}")),
    }
    if (
        authority["module_blob_matches"] is not True
        or authority["clean"] is not True
        or authority["upstream_ahead"]
        or authority["upstream_behind"]
    ):
        raise Plan0065D1Error("D1 repository authority is not clean and upstream-even.")
    return authority


def build_acoustic_safety_policy(
    *,
    threshold_application: Mapping[str, Any],
    score_matrix: Mapping[str, Any],
    threshold_application_file_sha256: str,
    score_matrix_file_sha256: str,
) -> dict[str, Any]:
    """Derive a boundary guard from frozen calibration genuine support."""

    if threshold_application.get("score_matrix_sha256") != score_matrix_file_sha256:
        raise Plan0065D1Error("Calibration threshold authority lost its score binding.")
    trials = list(score_matrix.get("trials") or [])
    floors = []
    excluded = []
    for unit in sorted(
        (
            item
            for item in threshold_application.get("thresholds") or []
            if item.get("method_id") == "no_enhancement"
        ),
        key=lambda item: str(item.get("candidate_id") or ""),
    ):
        candidate_id = str(unit.get("candidate_id") or "")
        metrics = unit.get("metrics") or {}
        threshold = float(unit.get("threshold"))
        if (
            threshold >= 1.0
            or int(metrics.get("false_reject_count") or 0) != 0
            or int(metrics.get("genuine_trial_count") or 0) < 1
        ):
            excluded.append(
                {
                    "candidate_id": candidate_id,
                    "reason_code": "calibration_did_not_accept_genuine_trials",
                }
            )
            continue
        genuine_scores = [
            float(item["score"])
            for item in trials
            if item.get("candidate_id") == candidate_id
            and item.get("method_id") == "no_enhancement"
            and item.get("expected_match") is True
        ]
        if len(genuine_scores) != int(metrics.get("genuine_trial_count") or 0):
            raise Plan0065D1Error("Calibration genuine denominator drifted.")
        minimum_genuine = min(genuine_scores)
        if minimum_genuine <= threshold:
            raise Plan0065D1Error("A usable calibration model lacks genuine separation.")
        minimum_safe = threshold + (
            (minimum_genuine - threshold)
            * SAFETY_NUMERATOR
            / SAFETY_DENOMINATOR
        )
        floors.append(
            {
                "candidate_id": candidate_id,
                "threshold": threshold,
                "minimum_calibration_genuine_score": minimum_genuine,
                "minimum_safe_score": minimum_safe,
            }
        )
    if len(floors) < 2:
        raise Plan0065D1Error("Fewer than two calibrated models support D1.")
    return _content_addressed(
        {
            "schema_version": POLICY_SCHEMA,
            "status": "frozen_calibration_boundary_and_probe_purity_guard",
            "derivation": (
                "threshold_plus_two_thirds_of_minimum_calibration_genuine_surplus"
            ),
            "safety_ratio": {
                "numerator": SAFETY_NUMERATOR,
                "denominator": SAFETY_DENOMINATOR,
            },
            "required_safe_model_count": 2,
            "required_probe_hash_match": True,
            "required_source_hash_match": True,
            "maximum_other_speaker_overlap_seconds": 0.0,
            "model_floors": floors,
            "excluded_models": excluded,
            "threshold_application_file_sha256": threshold_application_file_sha256,
            "score_matrix_file_sha256": score_matrix_file_sha256,
            "action_counts": dict(ACTION_COUNTS),
        }
    )


def apply_acoustic_safety_policy(
    slot: Mapping[str, Any], policy: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply the successor guard without changing underlying score evidence."""

    audit = slot.get("probe_audit") or {}
    prior_status = str(slot.get("status") or "unavailable")
    result = dict(slot)
    result["prior_status"] = prior_status
    result["policy_content_sha256"] = policy.get("content_sha256")
    if audit.get("source_hash_matches") is not True:
        result.update(
            status="unavailable",
            reason_code="source_hash_mismatch",
            candidate_person_id=None,
            candidate_acoustic_subject_id=None,
        )
        return result
    if audit.get("probe_hash_matches") is not True:
        result.update(
            status="unavailable",
            reason_code="probe_hash_mismatch",
            candidate_person_id=None,
            candidate_acoustic_subject_id=None,
        )
        return result
    if float(audit.get("other_speaker_overlap_seconds") or 0.0) > float(
        policy["maximum_other_speaker_overlap_seconds"]
    ):
        result.update(
            status="review",
            reason_code="diarization_overlap_guard",
            candidate_person_id=None,
            candidate_acoustic_subject_id=None,
        )
        return result
    if prior_status != "candidate":
        result["candidate_person_id"] = None
        result["candidate_acoustic_subject_id"] = None
        return result

    person_id = slot.get("candidate_person_id")
    floors = {item["candidate_id"]: item for item in policy["model_floors"]}
    safe_rows = [
        item
        for item in slot.get("model_rows") or []
        if item.get("candidate_id") in floors
        and item.get("threshold_pass") is True
        and item.get("binding_eligible") is True
        and item.get("top_canonical_person_id") == person_id
        and float(item.get("top_score"))
        >= float(floors[str(item["candidate_id"])]["minimum_safe_score"])
    ]
    result["safe_supporting_model_count"] = len(safe_rows)
    result["safe_supporting_model_ids"] = sorted(
        str(item["candidate_id"]) for item in safe_rows
    )
    if len(safe_rows) >= int(policy["required_safe_model_count"]):
        result["status"] = "candidate"
        result["reason_code"] = "multi_model_calibration_buffer_support"
    else:
        result.update(
            status="review",
            reason_code="calibration_boundary_guard",
            candidate_person_id=None,
            candidate_acoustic_subject_id=None,
        )
    return result


def evaluate_development_gate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    original_correct = sum(
        item.get("gold") == "correct" and item.get("before") == "candidate"
        for item in rows
    )
    original_wrong = sum(
        item.get("gold") == "wrong" and item.get("before") == "candidate"
        for item in rows
    )
    retained_correct = sum(
        item.get("gold") == "correct" and item.get("after") == "candidate"
        for item in rows
    )
    retained_wrong = sum(
        item.get("gold") == "wrong" and item.get("after") == "candidate"
        for item in rows
    )
    new_wrong = sum(
        item.get("gold") == "wrong"
        and item.get("before") != "candidate"
        and item.get("after") == "candidate"
        for item in rows
    )
    passed = (
        original_correct == 11
        and original_wrong == 1
        and retained_correct >= 10
        and retained_wrong == 0
        and new_wrong == 0
    )
    return {
        "original_correct_candidate_count": original_correct,
        "original_wrong_candidate_count": original_wrong,
        "retained_correct_candidate_count": retained_correct,
        "retained_wrong_candidate_count": retained_wrong,
        "new_wrong_candidate_count": new_wrong,
        "minimum_required_correct_retained": 10,
        "passed": passed,
        "terminal_status": "d1_pass" if passed else "acoustic_recovery_failed",
    }


def _selected_intervals(
    transcript: Mapping[str, Any], speaker: str
) -> list[tuple[float, float]]:
    remaining = float(p1.MAX_PROBE_SECONDS)
    intervals = []
    for utterance in transcript.get("utterances") or []:
        if not isinstance(utterance, Mapping) or str(utterance.get("speaker")) != speaker:
            continue
        start = max(0.0, float(utterance.get("start") or 0.0) / 1000.0)
        end = max(start, float(utterance.get("end") or 0.0) / 1000.0)
        duration = min(remaining, end - start)
        if duration > 0:
            intervals.append((start, start + duration))
            remaining -= duration
        if remaining <= 0:
            break
    return intervals


def _other_speaker_overlap_seconds(
    transcript: Mapping[str, Any], speaker: str, selected: Sequence[tuple[float, float]]
) -> float:
    other = []
    for utterance in transcript.get("utterances") or []:
        if not isinstance(utterance, Mapping) or str(utterance.get("speaker")) == speaker:
            continue
        start = max(0.0, float(utterance.get("start") or 0.0) / 1000.0)
        end = max(start, float(utterance.get("end") or 0.0) / 1000.0)
        other.append((start, end))
    return sum(
        max(0.0, min(left_end, right_end) - max(left_start, right_start))
        for left_start, left_end in selected
        for right_start, right_end in other
    )


def probe_authority_is_exact(audit: Mapping[str, Any]) -> bool:
    """Accept container reserialization only when acoustic derivation is exact."""

    return all(
        audit.get(key) is True
        for key in (
            "source_hash_matches",
            "probe_hash_matches",
            "probe_duration_matches",
        )
    )


def audit_probe(
    *,
    recording: Mapping[str, Any],
    slot: Mapping[str, Any],
) -> tuple[dict[str, Any], array]:
    transcript_path = Path(recording["transcript_artifact"]["path"])
    source_path = Path(recording["source_media_artifact"]["path"])
    transcript = _read_object(transcript_path)
    source_matches = sha256_file(source_path) == recording["source_media_sha256"]
    transcript_matches = sha256_file(transcript_path) == recording["artifact_sha256"]
    decoded = p1._decode(source_path)
    speaker = str(slot["speaker_label"])
    probe = p1._slot_probe(transcript, speaker, decoded)
    probe_sha256 = hashlib.sha256(array("f", probe).tobytes()).hexdigest()
    selected = _selected_intervals(transcript, speaker)
    audit = {
        "source_hash_matches": source_matches,
        "transcript_hash_matches": transcript_matches,
        "transcript_expected_file_sha256": recording["artifact_sha256"],
        "transcript_current_file_sha256": sha256_file(transcript_path),
        "probe_hash_matches": probe_sha256 == slot["probe_sha256"],
        "probe_duration_matches": abs(
            len(probe) / p1.SAMPLE_RATE - float(slot["probe_duration_seconds"])
        )
        < 1e-9,
        "selected_utterance_count": len(selected),
        "other_speaker_overlap_seconds": _other_speaker_overlap_seconds(
            transcript, speaker, selected
        ),
    }
    return audit, probe


def _temporal_audit(
    *,
    candidate_rows: Sequence[tuple[Mapping[str, Any], Mapping[str, Any], array]],
    profiles: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    adapters = {
        key: p1._CachingAdapter(value)
        for key, value in verification.adapter_registry().items()
    }
    rows = []
    for outcome, slot, probe in candidate_rows:
        midpoint = len(probe) // 2
        halves = (array("f", probe[:midpoint]), array("f", probe[midpoint:]))
        half_rows = []
        for index, half in enumerate(halves, start=1):
            scored = p1._score_slot(
                document_id=str(slot["speaker_ref"]).split("::", 1)[0],
                speaker=f"temporal-half-{index}",
                probe=half,
                profiles=profiles,
                thresholds=thresholds,
                adapters=adapters,
                score_fn=verification.score_profile,
                profile_root=p1.DEFAULT_PROFILE_ROOT,
                reference_root=p1.DEFAULT_REFERENCE_ROOT,
            )
            half_rows.append(
                {
                    "status": scored["status"],
                    "candidate_matches_aggregate": (
                        scored.get("candidate_person_id")
                        == slot.get("candidate_person_id")
                    ),
                    "top_rank_match_count": sum(
                        item.get("top_canonical_person_id")
                        == slot.get("candidate_person_id")
                        for item in scored["model_rows"]
                    ),
                }
            )
        rows.append(
            {
                "speaker_ref": slot["speaker_ref"],
                "gold_outcome": outcome["gold"],
                "halves": half_rows,
                "both_halves_same_candidate": all(
                    item["status"] == "candidate"
                    and item["candidate_matches_aggregate"]
                    for item in half_rows
                ),
                "third_model_half_support": any(
                    item["top_rank_match_count"] == 3 for item in half_rows
                ),
            }
        )
    correct_rows = [item for item in rows if item["gold_outcome"] == "correct"]
    wrong_rows = [item for item in rows if item["gold_outcome"] == "wrong"]
    return _content_addressed(
        {
            "status": "diagnostic_only_not_a_policy_input",
            "rows": rows,
            "summary": {
                "candidate_count": len(rows),
                "correct_count": len(correct_rows),
                "wrong_count": len(wrong_rows),
                "correct_temporally_stable_count": sum(
                    item["both_halves_same_candidate"] for item in correct_rows
                ),
                "wrong_temporally_stable_count": sum(
                    item["both_halves_same_candidate"] for item in wrong_rows
                ),
                "correct_third_model_half_support_count": sum(
                    item["third_model_half_support"] for item in correct_rows
                ),
                "wrong_third_model_half_support_count": sum(
                    item["third_model_half_support"] for item in wrong_rows
                ),
            },
            "action_counts": dict(ACTION_COUNTS),
        }
    )


def _paths(runtime_root: Path, policy_sha256: str) -> dict[str, Path]:
    root = runtime_root.expanduser().absolute()
    run = root / f"d1-{policy_sha256[:24]}"
    return {
        "root": root,
        "run": run,
        "policy": run / "policy.json",
        "evidence": run / "private-development-evidence.json",
        "receipt": run / "receipt.json",
    }


def _authorities(
    *,
    d0_manifest_path: Path,
    d0_receipt_path: Path,
    threshold_application_path: Path,
    score_matrix_path: Path,
) -> dict[str, Any]:
    d0_manifest = read_private_object(d0_manifest_path.expanduser())
    d0_receipt = read_private_object(d0_receipt_path.expanduser())
    _validate_content(d0_manifest, label="D0 manifest")
    _validate_content(d0_receipt, label="D0 receipt")
    if (
        d0_manifest.get("content_sha256") != D0_MANIFEST_SHA256
        or d0_receipt.get("content_sha256") != D0_RECEIPT_SHA256
        or any((d0_receipt.get("action_counts") or {}).values())
    ):
        raise Plan0065D1Error("D0 authority is not the frozen zero-effect packet.")
    threshold_path = threshold_application_path.expanduser()
    matrix_path = score_matrix_path.expanduser()
    if (
        sha256_file(threshold_path) != CALIBRATION_THRESHOLD_FILE_SHA256
        or sha256_file(matrix_path) != CALIBRATION_SCORE_MATRIX_FILE_SHA256
    ):
        raise Plan0065D1Error("Frozen calibration authority drifted.")
    threshold_application = _read_object(threshold_path)
    score_matrix = _read_object(matrix_path)
    policy = build_acoustic_safety_policy(
        threshold_application=threshold_application,
        score_matrix=score_matrix,
        threshold_application_file_sha256=sha256_file(threshold_path),
        score_matrix_file_sha256=sha256_file(matrix_path),
    )
    return {
        "d0_manifest": d0_manifest,
        "d0_receipt": d0_receipt,
        "threshold_application": threshold_application,
        "score_matrix": score_matrix,
        "policy": policy,
    }


def execute_d1(
    *,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    d0_manifest_path: Path = DEFAULT_D0_MANIFEST,
    d0_receipt_path: Path = DEFAULT_D0_RECEIPT,
    threshold_application_path: Path = DEFAULT_THRESHOLD_APPLICATION,
    score_matrix_path: Path = DEFAULT_SCORE_MATRIX,
) -> dict[str, Any]:
    repository = repository_authority()
    authority = _authorities(
        d0_manifest_path=d0_manifest_path,
        d0_receipt_path=d0_receipt_path,
        threshold_application_path=threshold_application_path,
        score_matrix_path=score_matrix_path,
    )
    policy = authority["policy"]
    paths = _paths(runtime_root, policy["content_sha256"])
    if paths["receipt"].exists():
        return replay_d1(
            policy_content_sha256=policy["content_sha256"],
            runtime_root=runtime_root,
            d0_manifest_path=d0_manifest_path,
            d0_receipt_path=d0_receipt_path,
            threshold_application_path=threshold_application_path,
            score_matrix_path=score_matrix_path,
        )
    if paths["run"].exists():
        raise Plan0065D1Error("A partial D1 runtime directory exists.")

    plan64_paths = d0._plan0064_paths(d0.DEFAULT_PLAN0064_ROOT)
    p0_manifest = read_private_object(plan64_paths["p0"] / "private-manifest.json")
    p1_evidence = read_private_object(
        plan64_paths["p1"] / "private-acoustic-evidence.json"
    )
    measurement = read_private_object(plan64_paths["measurement"] / "measurement.json")
    if (
        p1_evidence.get("content_sha256") != d0.P1_CONTENT_SHA256
        or measurement.get("content_sha256") != d0.MEASUREMENT_SHA256
    ):
        raise Plan0065D1Error("Plan 0064 acoustic/gold authority drifted.")
    p0_profile_matrix = sorted(
        (
            item["candidate_id"],
            item["profile_id"],
            item["artifact"]["sha256"],
            item.get("canonical_person_id"),
            bool(item["identity_candidate_eligible"]),
        )
        for item in p0_manifest["profile_inventory"]["active_profiles"]
    )
    d0_profile_matrix = sorted(
        (
            item["candidate_id"],
            item["profile_id"],
            item["artifact"]["sha256"],
            item.get("canonical_person_id"),
            bool(item["identity_candidate_eligible"]),
        )
        for item in authority["d0_manifest"]["current_profile_inventory"]
        ["profile_inventory"]["active_profiles"]
    )
    if p0_profile_matrix != d0_profile_matrix:
        raise Plan0065D1Error("The Plan 0064 scoring profile matrix drifted before D1.")
    selected = {
        item["document_id"]: item
        for item in p0_manifest["evaluation_cohort"]["considered"]
        if item.get("disposition") == "selected_evaluation_candidate"
    }
    prior_development_hashes = set(
        p0_manifest["reference_inventory"]["development_recording_hashes"]
    )
    selected_source_hashes = {
        item["source_media_sha256"] for item in selected.values()
    }
    source_overlap = sorted(prior_development_hashes & selected_source_hashes)
    if source_overlap:
        raise Plan0065D1Error("A Plan 0064 probe overlaps profile development audio.")
    acoustic_gold = {
        row["speaker_ref"]: next(
            condition
            for condition in row["conditions"]
            if condition["condition"] == "acoustic"
        )
        for row in measurement["rows"]
    }
    development_rows = []
    temporal_inputs = []
    purity_counts: Counter[str] = Counter()
    for evidence_recording in p1_evidence["recordings"]:
        recording = selected[evidence_recording["document_id"]]
        for slot in evidence_recording["speaker_slots"]:
            audit, probe = audit_probe(recording=recording, slot=slot)
            if not probe_authority_is_exact(audit):
                raise Plan0065D1Error("A Plan 0064 probe failed exact reproduction.")
            purity_counts[
                "overlap" if audit["other_speaker_overlap_seconds"] > 0 else "clean"
            ] += 1
            purity_counts[
                "transcript_file_exact"
                if audit["transcript_hash_matches"]
                else "transcript_container_drift_probe_exact"
            ] += 1
            original = {**slot, "probe_audit": audit}
            corrected = apply_acoustic_safety_policy(original, policy)
            gold = acoustic_gold[slot["speaker_ref"]]
            outcome = (
                "wrong"
                if gold["wrong_candidate"]
                else "correct"
                if gold["correct_candidate"]
                else "other"
            )
            row = {
                "speaker_ref": slot["speaker_ref"],
                "gold": outcome,
                "before": slot["status"],
                "after": corrected["status"],
                "reason_code": corrected["reason_code"],
                "probe_audit": audit,
                "safe_supporting_model_count": corrected.get(
                    "safe_supporting_model_count", 0
                ),
            }
            development_rows.append(row)
            if slot["status"] == "candidate":
                temporal_inputs.append((row, slot, probe))
    gate_rows = [item for item in development_rows if item["before"] == "candidate"]
    gate = evaluate_development_gate(gate_rows)
    if gate["passed"] is not True:
        raise Plan0065D1Error("The frozen D1 development gate did not pass.")
    thresholds = {
        item["candidate_id"]: float(item["threshold"])
        for item in p1_evidence["threshold_authority"]["units"]
    }
    temporal = _temporal_audit(
        candidate_rows=temporal_inputs,
        profiles=p0_manifest["profile_inventory"]["active_profiles"],
        thresholds=thresholds,
    )
    evidence = _content_addressed(
        {
            "schema_version": EVIDENCE_SCHEMA,
            "status": "d1_development_gate_passed",
            "repository_authority": repository,
            "d0_manifest_content_sha256": D0_MANIFEST_SHA256,
            "d0_receipt_content_sha256": D0_RECEIPT_SHA256,
            "p1_evidence_content_sha256": d0.P1_CONTENT_SHA256,
            "measurement_content_sha256": d0.MEASUREMENT_SHA256,
            "policy": policy,
            "development_rows": development_rows,
            "development_row_set_sha256": _hash(development_rows),
            "probe_purity_counts": dict(sorted(purity_counts.items())),
            "profile_and_source_audit": {
                "profile_matrix_matches_d0": True,
                "active_profile_count": len(p0_profile_matrix),
                "probe_to_profile_development_recording_overlap_count": 0,
            },
            "temporal_diagnostic": temporal,
            "development_gate": gate,
            "hypothesis_dispositions": {
                "probe_hash_or_source_drift": "not_observed",
                "transcript_container_drift": (
                    "observed_but_all_derived_probe_hashes_remain_exact"
                ),
                "diarization_interval_overlap": "guarded_not_observed_in_candidates",
                "temporal_candidate_instability": "rejected_as_discriminator",
                "third_model_temporal_support": "rejected_for_low_correct_retention",
                "threshold_only": "rejected_without_calibration_buffer",
                "calibration_boundary": "supported_and_frozen",
            },
            "execution_counts": {
                "local_biometric_half_probe_count": len(temporal_inputs) * 2,
                "provider_model_turn_count": 0,
            },
            "action_counts": dict(ACTION_COUNTS),
        }
    )
    ensure_private_tree(paths["root"], paths["run"])
    write_immutable_private_json(paths["policy"], policy)
    write_immutable_private_json(paths["evidence"], evidence)
    receipt = _content_addressed(
        {
            "schema_version": RECEIPT_SCHEMA,
            "status": "d1_pass_zero_effect",
            "policy_content_sha256": policy["content_sha256"],
            "policy_file_sha256": sha256_file(paths["policy"]),
            "evidence_content_sha256": evidence["content_sha256"],
            "evidence_file_sha256": sha256_file(paths["evidence"]),
            "development_gate": gate,
            "probe_purity_counts": evidence["probe_purity_counts"],
            "temporal_summary": temporal["summary"],
            "execution_counts": evidence["execution_counts"],
            "action_counts": dict(ACTION_COUNTS),
        }
    )
    write_immutable_private_json(paths["receipt"], receipt)
    return {
        **receipt,
        "private_policy_path": str(paths["policy"]),
        "private_evidence_path": str(paths["evidence"]),
        "private_receipt_path": str(paths["receipt"]),
        "idempotent_replay": False,
    }


def replay_d1(
    *,
    policy_content_sha256: str,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    d0_manifest_path: Path = DEFAULT_D0_MANIFEST,
    d0_receipt_path: Path = DEFAULT_D0_RECEIPT,
    threshold_application_path: Path = DEFAULT_THRESHOLD_APPLICATION,
    score_matrix_path: Path = DEFAULT_SCORE_MATRIX,
) -> dict[str, Any]:
    repository_authority()
    authority = _authorities(
        d0_manifest_path=d0_manifest_path,
        d0_receipt_path=d0_receipt_path,
        threshold_application_path=threshold_application_path,
        score_matrix_path=score_matrix_path,
    )
    if authority["policy"]["content_sha256"] != policy_content_sha256:
        raise Plan0065D1Error("The requested D1 policy is not current authority.")
    paths = _paths(runtime_root, policy_content_sha256)
    for path in (paths["policy"], paths["evidence"], paths["receipt"]):
        require_private_file(path, paths["root"])
    policy = read_private_object(paths["policy"])
    evidence = read_private_object(paths["evidence"])
    receipt = read_private_object(paths["receipt"])
    _validate_content(policy, label="D1 policy")
    _validate_content(evidence, label="D1 evidence")
    _validate_content(receipt, label="D1 receipt")
    if (
        policy != authority["policy"]
        or receipt.get("policy_content_sha256") != policy_content_sha256
        or receipt.get("policy_file_sha256") != sha256_file(paths["policy"])
        or receipt.get("evidence_content_sha256") != evidence.get("content_sha256")
        or receipt.get("evidence_file_sha256") != sha256_file(paths["evidence"])
        or evidence.get("development_gate", {}).get("passed") is not True
        or any((receipt.get("action_counts") or {}).values())
    ):
        raise Plan0065D1Error("Frozen D1 evidence drifted.")
    return {
        **receipt,
        "private_policy_path": str(paths["policy"]),
        "private_evidence_path": str(paths["evidence"]),
        "private_receipt_path": str(paths["receipt"]),
        "idempotent_replay": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("execute", "replay"))
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--policy-content-sha256")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.mode == "execute":
        result = execute_d1(runtime_root=args.runtime_root)
    else:
        if not args.policy_content_sha256:
            raise SystemExit("replay requires --policy-content-sha256")
        result = replay_d1(
            policy_content_sha256=args.policy_content_sha256,
            runtime_root=args.runtime_root,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
