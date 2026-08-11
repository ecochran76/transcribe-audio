"""Reconstruct and freeze Plan 0071 joined/residual development evidence."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import speaker_identity_plan0064_p2 as plan0064_p2
import speaker_identity_plan0064_p3 as plan0064_p3
import speaker_identity_plan0065_d1 as plan0065_d1
import speaker_identity_plan0069_a2 as plan0069_a2
import speaker_identity_plan0071_d0 as d0
from acoustic_audio_derivatives import (
    ensure_private_tree,
    read_private_object,
    sha256_file,
    write_immutable_private_json,
)


SCHEMA_VERSION = "transcribe-audio.plan0071-d1-resolution.v1"
MEASUREMENT_SCHEMA_VERSION = "transcribe-audio.plan0071-d1-measurement.v1"
RECEIPT_SCHEMA_VERSION = "transcribe-audio.plan0071-d1-receipt.v1"
DEFAULT_RUNTIME_ROOT = d0.DEFAULT_RUNTIME_ROOT
D0_ACTIVATION_CONTENT_SHA256 = (
    "27b011dcce9b3df6922ae0d1f91b077249c2ca991e804e0ada0d40c5713ac931"
)
D0_RECEIPT_CONTENT_SHA256 = (
    "ef0c0e8af6b54701668847ec22a407e71226b4aaf4a26e97b5a9ff14c6b79a69"
)
CONDITIONS = ("acoustic", "context", "combined", "residual_policy")
EXPECTED_MEASUREMENT = {
    "recording_count": 12,
    "speaker_slot_count": 39,
    "condition_counts": {
        "acoustic": {
            "correct_candidate_count": 10,
            "wrong_candidate_count": 0,
            "abstained_slot_count": 29,
        },
        "context": {
            "correct_candidate_count": 5,
            "wrong_candidate_count": 1,
            "abstained_slot_count": 33,
        },
        "combined": {
            "correct_candidate_count": 5,
            "wrong_candidate_count": 0,
            "abstained_slot_count": 34,
        },
        "residual_policy": {
            "correct_candidate_count": 5,
            "wrong_candidate_count": 0,
            "abstained_slot_count": 34,
        },
    },
    "correct_pillar_agreement_count": 5,
    "wrong_combined_candidate_count": 0,
    "wrong_context_candidate_safely_unaccepted_count": 1,
    "actual_residual_acceptance_count": 0,
}
EFFECT_COUNTS = dict(d0.EFFECT_COUNTS)


class Plan0071D1Error(ValueError):
    """Raised when deterministic joined evidence is incomplete or drifts."""


def _hash(value: Any) -> str:
    return d0._hash(value)


def _content(value: Mapping[str, Any]) -> dict[str, Any]:
    return d0._content(value)


def _validate_content(value: Mapping[str, Any], label: str) -> None:
    try:
        d0._validate_content(value, label)
    except d0.Plan0071D0Error as exc:
        raise Plan0071D1Error(str(exc)) from exc


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], check=False, capture_output=True, text=True
    )
    if result.returncode:
        raise Plan0071D1Error(result.stderr.strip() or "Git authority read failed.")
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
    authority = {
        "module_name": relative,
        "module_commit": commit,
        "module_sha256": hashlib.sha256(module.read_bytes()).hexdigest(),
        "module_blob_matches": committed.returncode == 0
        and hashlib.sha256(module.read_bytes()).hexdigest()
        == hashlib.sha256(committed.stdout).hexdigest(),
        "clean": not _git("status", "--porcelain=v1"),
        "upstream_ahead": int(_git("rev-list", "--count", f"{upstream}..HEAD")),
        "upstream_behind": int(_git("rev-list", "--count", f"HEAD..{upstream}")),
    }
    if authority["module_blob_matches"] is not True or (
        require_clean
        and (
            authority["clean"] is not True
            or authority["upstream_ahead"]
            or authority["upstream_behind"]
        )
    ):
        raise Plan0071D1Error("D1 source authority is not acceptable.")
    return authority


def _artifact_path(manifest: Mapping[str, Any], key: str) -> Path:
    binding = (manifest.get("artifact_bindings") or {}).get(key)
    if not isinstance(binding, Mapping):
        raise Plan0071D1Error(f"D0 lacks artifact binding {key}.")
    path = Path(str(binding.get("path") or "")).expanduser().resolve()
    if sha256_file(path) != binding.get("file_sha256"):
        raise Plan0071D1Error(f"D0 artifact binding drifted: {key}.")
    return path


def _plan0064_review_authority(plan0065_d0: Mapping[str, Any]) -> Path:
    bindings = plan0065_d0.get("plan0064_authority", {}).get("artifact_bindings")
    matches = [
        Path(str(item.get("path") or "")).expanduser().resolve()
        for item in bindings or []
        if isinstance(item, Mapping)
        and item.get("label") == "Plan 0064 P4 review authority"
    ]
    if len(matches) != 1:
        raise Plan0071D1Error("Plan 0064 review filename authority is ambiguous.")
    expected = next(
        item
        for item in bindings
        if item.get("label") == "Plan 0064 P4 review authority"
    )
    if sha256_file(matches[0]) != expected.get("file_sha256"):
        raise Plan0071D1Error("Plan 0064 review filename authority drifted.")
    return matches[0]


def _filename_map(review: Mapping[str, Any]) -> dict[str, str]:
    by_document: dict[str, set[str]] = {}
    for case in review.get("cases") or []:
        if not isinstance(case, Mapping):
            raise Plan0071D1Error("Plan 0064 review contains an invalid case.")
        document_id = str(case.get("document_id") or "")
        filename = str(case.get("recording_filename") or "")
        if not document_id or not filename:
            raise Plan0071D1Error("Plan 0064 review filename is incomplete.")
        by_document.setdefault(document_id, set()).add(filename)
    if len(by_document) != 12 or any(len(values) != 1 for values in by_document.values()):
        raise Plan0071D1Error("Plan 0064 review filename denominator drifted.")
    return {key: next(iter(values)) for key, values in by_document.items()}


def _corrected_acoustic_recordings(
    *,
    p1_evidence: Mapping[str, Any],
    d1_evidence: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = {
        str(item.get("speaker_ref") or ""): item
        for item in d1_evidence.get("development_rows") or []
        if isinstance(item, Mapping)
    }
    recordings = []
    for recording in p1_evidence.get("recordings") or []:
        corrected_slots = []
        for slot in recording.get("speaker_slots") or []:
            row = rows.get(str(slot.get("speaker_ref") or ""))
            if not isinstance(row, Mapping):
                raise Plan0071D1Error("Corrected acoustic row is missing.")
            corrected = plan0065_d1.apply_acoustic_safety_policy(
                {**slot, "probe_audit": dict(row.get("probe_audit") or {})},
                policy,
            )
            if (
                corrected.get("status") != row.get("after")
                or corrected.get("reason_code") != row.get("reason_code")
                or corrected.get("safe_supporting_model_count", 0)
                != row.get("safe_supporting_model_count", 0)
            ):
                raise Plan0071D1Error("Corrected acoustic policy replay drifted.")
            corrected_slots.append(corrected)
        recordings.append({**recording, "speaker_slots": corrected_slots})
    if len(recordings) != 12 or sum(
        len(item.get("speaker_slots") or []) for item in recordings
    ) != 39:
        raise Plan0071D1Error("Corrected acoustic denominator drifted.")
    return recordings


def _context_cases(
    *,
    plan0065_case_root: Path,
    plan0069_case_root: Path,
    canonical_people: set[str],
    filenames: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    plan0065_paths = sorted(plan0065_case_root.glob("*.json"))
    if len(plan0065_paths) != 12:
        raise Plan0071D1Error("Plan 0065 context denominator drifted.")
    contexts = {}
    for path in plan0065_paths:
        case = read_private_object(path)
        _validate_content(case, f"Plan 0065 context case {path.stem}")
        contexts[path.stem] = dict(case)
    plan0069_paths = sorted(plan0069_case_root.glob("*.json"))
    if len(plan0069_paths) != 6:
        raise Plan0071D1Error("Plan 0069 context denominator drifted.")
    for path in plan0069_paths:
        case = read_private_object(path)
        _validate_content(case, f"Plan 0069 context case {path.stem}")
        document_id = path.stem
        prior = contexts[document_id]
        plan0069_filename = str(case.get("original_recording_filename") or "")
        if not plan0069_filename:
            raise Plan0071D1Error("Plan 0069 source artifact filename is missing.")
        labels = [str(item["speaker_label"]) for item in prior["speaker_slots"]]
        validated = case.get("validated_readout") or {}
        slots = plan0064_p2._proposal_slot_rows(
            document_id=document_id,
            speaker_labels=labels,
            prediction={"proposals": list(validated.get("speaker_assignments") or [])},
            canonical_people=canonical_people,
        )
        contexts[document_id] = {
            **prior,
            "status": "plan0069_normalized_context_replayed",
            "speaker_slots": slots,
            "prediction": dict(validated),
            "provider_failures": [],
            "original_recording_filename": filenames[document_id],
            "plan0069_source_artifact_filename": plan0069_filename,
            "plan0069_case_content_sha256": case["content_sha256"],
        }
    return contexts


def measure_resolutions(
    recordings: Sequence[Mapping[str, Any]], gold: Mapping[str, Any]
) -> dict[str, Any]:
    gold_rows = {
        str(item.get("speaker_ref") or ""): item
        for item in gold.get("decisions") or []
        if isinstance(item, Mapping)
    }
    counts = {
        condition: {
            "correct_candidate_count": 0,
            "wrong_candidate_count": 0,
            "abstained_slot_count": 0,
        }
        for condition in CONDITIONS
    }
    reason_counts: dict[str, Counter[str]] = {
        condition: Counter() for condition in CONDITIONS
    }
    wrong_context_safely_unaccepted = 0
    for recording in recordings:
        for slot in recording.get("speaker_slots") or []:
            speaker_ref = str(slot.get("speaker_ref") or "")
            decision = gold_rows.get(speaker_ref)
            if not isinstance(decision, Mapping):
                raise Plan0071D1Error(f"Human gold is missing {speaker_ref}.")
            expected = (
                str(decision.get("person_id") or "")
                if decision.get("decision") == "canonical_person"
                else ""
            )
            for condition in CONDITIONS:
                prediction = slot.get(condition) or {}
                disposition = str(prediction.get("disposition") or "")
                reason_counts[condition][
                    str(prediction.get("reason_code") or "missing")
                ] += 1
                if disposition != "candidate":
                    counts[condition]["abstained_slot_count"] += 1
                elif expected and prediction.get("candidate_person_id") == expected:
                    counts[condition]["correct_candidate_count"] += 1
                else:
                    counts[condition]["wrong_candidate_count"] += 1
            context = slot.get("context") or {}
            combined = slot.get("combined") or {}
            residual = slot.get("residual_policy") or {}
            if (
                context.get("disposition") == "candidate"
                and not (
                    expected and context.get("candidate_person_id") == expected
                )
                and combined.get("disposition") != "candidate"
                and residual.get("disposition") != "candidate"
            ):
                wrong_context_safely_unaccepted += 1
    recording_count = len(recordings)
    speaker_slot_count = sum(
        len(item.get("speaker_slots") or []) for item in recordings
    )
    pillar_count = sum(
        reason_counts["combined"].get(reason, 0)
        for reason in (
            "pillar_agreement",
            "pillar_agreement_same_person_multi_label",
        )
    )
    residual_count = reason_counts["residual_policy"].get(
        "two_known_plus_one_independently_supported_residual", 0
    )
    core = {
        "recording_count": recording_count,
        "speaker_slot_count": speaker_slot_count,
        "condition_counts": counts,
        "correct_pillar_agreement_count": pillar_count,
        "wrong_combined_candidate_count": counts["combined"][
            "wrong_candidate_count"
        ],
        "wrong_context_candidate_safely_unaccepted_count": (
            wrong_context_safely_unaccepted
        ),
        "actual_residual_acceptance_count": residual_count,
    }
    passed = core == EXPECTED_MEASUREMENT
    return _content(
        {
            "schema_version": MEASUREMENT_SCHEMA_VERSION,
            "status": (
                "joined_gate_passed_residual_gap_proven"
                if passed
                else "joined_gate_not_reproduced"
            ),
            **core,
            "reason_counts": {
                key: dict(sorted(value.items()))
                for key, value in reason_counts.items()
            },
            "passed": passed,
            "supplemental_development_required": passed and residual_count == 0,
            "fresh_evaluation_allowed": False,
            "effect_counts": dict(EFFECT_COUNTS),
        }
    )


def _build_evidence() -> tuple[dict[str, Any], dict[str, Any]]:
    d0_receipt = d0.replay_activation(runtime_root=DEFAULT_RUNTIME_ROOT)
    if (
        d0_receipt.get("activation_content_sha256")
        != D0_ACTIVATION_CONTENT_SHA256
        or d0_receipt.get("content_sha256") != D0_RECEIPT_CONTENT_SHA256
    ):
        raise Plan0071D1Error("D0 authority is not the frozen Plan 0071 packet.")
    d0_manifest = read_private_object(Path(str(d0_receipt["manifest_path"])))
    _validate_content(d0_manifest, "Plan 0071 D0 manifest")

    p65_d0_path = _artifact_path(d0_manifest, "plan0065_d0_manifest")
    p65_d1_policy_path = _artifact_path(d0_manifest, "plan0065_d1_policy")
    p65_d1_evidence_path = _artifact_path(d0_manifest, "plan0065_d1_evidence")
    p65_d2_receipt_path = _artifact_path(d0_manifest, "plan0065_d2_receipt")
    p69_a2_manifest_path = _artifact_path(d0_manifest, "plan0069_a2_manifest")
    p64_p1_path = _artifact_path(d0_manifest, "plan0064_p1_evidence")
    p64_gold_path = _artifact_path(d0_manifest, "plan0064_human_gold")

    p65_d0 = read_private_object(p65_d0_path)
    policy = read_private_object(p65_d1_policy_path)
    d1_evidence = read_private_object(p65_d1_evidence_path)
    p1_evidence = read_private_object(p64_p1_path)
    gold = read_private_object(p64_gold_path)
    for value, label in (
        (p65_d0, "Plan 0065 D0 manifest"),
        (policy, "Plan 0065 D1 policy"),
        (d1_evidence, "Plan 0065 D1 evidence"),
        (p1_evidence, "Plan 0064 P1 evidence"),
        (gold, "Plan 0064 human gold"),
    ):
        _validate_content(value, label)
    plan0065_d1.replay_d1(
        policy_content_sha256=str(policy["content_sha256"]),
        runtime_root=p65_d0_path.parents[1],
    )
    plan0069_a2.replay_a2(runtime_root=p69_a2_manifest_path.parents[1])

    review_path = _plan0064_review_authority(p65_d0)
    review = read_private_object(review_path)
    _validate_content(review, "Plan 0064 review authority")
    filenames = _filename_map(review)
    canonical_people = {
        str(item.get("person_id") or "")
        for item in p65_d0["current_profile_inventory"]["canonical_bindings"][
            "subject_bindings"
        ]
        if item.get("identity_candidate_eligible") is True and item.get("person_id")
    }
    if len(canonical_people) != 6:
        raise Plan0071D1Error("Canonical person allowlist drifted.")

    acoustic_recordings = _corrected_acoustic_recordings(
        p1_evidence=p1_evidence,
        d1_evidence=d1_evidence,
        policy=policy,
    )
    contexts = _context_cases(
        plan0065_case_root=p65_d2_receipt_path.parent / "cases",
        plan0069_case_root=p69_a2_manifest_path.parent / "cases",
        canonical_people=canonical_people,
        filenames=filenames,
    )
    resolved = []
    for acoustic in acoustic_recordings:
        document_id = str(acoustic["document_id"])
        resolution = plan0064_p3.resolve_conversation(
            acoustic, contexts[document_id]
        )
        resolved.append(
            {
                **resolution,
                "original_recording_filename": filenames[document_id],
                "context_authority": (
                    "plan0069_normalized_context"
                    if "plan0069_case_content_sha256" in contexts[document_id]
                    else "plan0065_d2_context"
                ),
            }
        )
    resolved.sort(key=lambda item: str(item["document_id"]))
    resolution = _content(
        {
            "schema_version": SCHEMA_VERSION,
            "status": "d1_joined_resolution_complete_zero_effect",
            "d0_activation_content_sha256": D0_ACTIVATION_CONTENT_SHA256,
            "policy_content_sha256": policy["content_sha256"],
            "human_gold_content_sha256": gold["content_sha256"],
            "recording_count": len(resolved),
            "speaker_slot_count": sum(
                len(item.get("speaker_slots") or []) for item in resolved
            ),
            "original_recording_filename_count": len(filenames),
            "recordings": resolved,
            "model_turn_count": 0,
            "effect_counts": dict(EFFECT_COUNTS),
        }
    )
    measurement = measure_resolutions(resolved, gold)
    if measurement.get("passed") is not True:
        raise Plan0071D1Error("D1 joined measurement did not reproduce exactly.")
    return resolution, measurement


def _paths(runtime_root: Path, resolution_sha256: str) -> dict[str, Path]:
    root = runtime_root.expanduser().resolve()
    run = root / f"d1-{resolution_sha256[:24]}"
    return {
        "root": root,
        "run": run,
        "resolution": run / "private-resolution.json",
        "measurement": run / "private-measurement.json",
        "receipt": run / "receipt.json",
    }


def execute_d1(*, runtime_root: Path = DEFAULT_RUNTIME_ROOT) -> dict[str, Any]:
    existing = list(runtime_root.expanduser().resolve().glob("d1-*/receipt.json"))
    if existing:
        return replay_d1(runtime_root=runtime_root)
    source_authority = _source_authority(require_clean=True)
    resolution, measurement = _build_evidence()
    paths = _paths(runtime_root, resolution["content_sha256"])
    if paths["run"].exists():
        raise Plan0071D1Error("A partial Plan 0071 D1 directory exists.")
    ensure_private_tree(paths["root"], paths["run"])
    write_immutable_private_json(paths["resolution"], resolution)
    write_immutable_private_json(paths["measurement"], measurement)
    receipt = _content(
        {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "status": "d1_joined_gate_passed_residual_gap_proven_zero_effect",
            "source_authority": source_authority,
            "d0_receipt_content_sha256": D0_RECEIPT_CONTENT_SHA256,
            "resolution_content_sha256": resolution["content_sha256"],
            "resolution_file_sha256": sha256_file(paths["resolution"]),
            "measurement_content_sha256": measurement["content_sha256"],
            "measurement_file_sha256": sha256_file(paths["measurement"]),
            "recording_count": 12,
            "speaker_slot_count": 39,
            "original_recording_filename_count": 12,
            "condition_counts": measurement["condition_counts"],
            "correct_pillar_agreement_count": 5,
            "actual_residual_acceptance_count": 0,
            "supplemental_development_required": True,
            "fresh_evaluation_allowed": False,
            "model_turn_count": 0,
            "effect_counts": dict(EFFECT_COUNTS),
        }
    )
    write_immutable_private_json(paths["receipt"], receipt)
    return {
        **receipt,
        "resolution_path": str(paths["resolution"]),
        "measurement_path": str(paths["measurement"]),
        "receipt_path": str(paths["receipt"]),
        "idempotent_replay": False,
    }


def replay_d1(*, runtime_root: Path = DEFAULT_RUNTIME_ROOT) -> dict[str, Any]:
    root = runtime_root.expanduser().resolve()
    receipts = list(root.glob("d1-*/receipt.json"))
    if len(receipts) != 1:
        raise Plan0071D1Error("Expected one Plan 0071 D1 receipt.")
    paths = {
        "receipt": receipts[0],
        "resolution": receipts[0].with_name("private-resolution.json"),
        "measurement": receipts[0].with_name("private-measurement.json"),
    }
    receipt = read_private_object(paths["receipt"])
    resolution = read_private_object(paths["resolution"])
    measurement = read_private_object(paths["measurement"])
    for value, label in (
        (receipt, "Plan 0071 D1 receipt"),
        (resolution, "Plan 0071 D1 resolution"),
        (measurement, "Plan 0071 D1 measurement"),
    ):
        _validate_content(value, label)
    expected_resolution, expected_measurement = _build_evidence()
    source = receipt.get("source_authority") or {}
    current_source = _source_authority(require_clean=False)
    if (
        resolution != expected_resolution
        or measurement != expected_measurement
        or receipt.get("resolution_content_sha256") != resolution["content_sha256"]
        or receipt.get("resolution_file_sha256") != sha256_file(paths["resolution"])
        or receipt.get("measurement_content_sha256") != measurement["content_sha256"]
        or receipt.get("measurement_file_sha256") != sha256_file(paths["measurement"])
        or source.get("module_sha256") != current_source.get("module_sha256")
        or receipt.get("effect_counts") != EFFECT_COUNTS
        or receipt.get("fresh_evaluation_allowed") is not False
    ):
        raise Plan0071D1Error("Plan 0071 D1 replay drifted.")
    return {
        **receipt,
        "resolution_path": str(paths["resolution"]),
        "measurement_path": str(paths["measurement"]),
        "receipt_path": str(paths["receipt"]),
        "idempotent_replay": True,
    }


if __name__ == "__main__":
    print(json.dumps(execute_d1(), indent=2, sort_keys=True))
