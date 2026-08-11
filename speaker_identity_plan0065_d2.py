#!/usr/bin/env python3
"""Plan 0065 D2 contextual evidence repair and bounded development run."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from acoustic_audio_derivatives import (
    canonical_artifact_hash,
    ensure_private_tree,
    read_private_object,
    require_private_file,
    sha256_file,
    write_immutable_private_json,
)
import app_intelligence_ledger
from speaker_evaluation_baseline import LocalSpeakerCaseRunner
import speaker_identity_plan0064_p2 as p2
import speaker_identity_plan0065_d0 as d0
import speaker_identity_plan0065_d1 as d1
from speaker_identity_preprocess import (
    CLUE_DISCOVERY_READOUT_SCHEMA_VERSION,
    build_clue_discovery_packet,
    validate_and_score_identity_evaluation,
    validate_clue_discovery_readout,
)


DEFAULT_RUNTIME_ROOT = d0.DEFAULT_RUNTIME_ROOT
DEFAULT_STATE_ROOT = Path("~/.local/state/transcribe-audio")
DEFAULT_API_BASE_URL = d0.DEFAULT_API_BASE_URL
D1_POLICY_SHA256 = "006c6770246619b011a390670824fbba8477443ed369cc4557a8ea96145548a6"
D1_RECEIPT_SHA256 = "7685aa856d50968b5f42026ce2f8b4bec87ec24ac24fbc352148bbef96f163ca"

ACTIVATION_SCHEMA = "transcribe-audio.plan0065-d2-activation.v1"
ACTIVATION_RECEIPT_SCHEMA = "transcribe-audio.plan0065-d2-activation-receipt.v1"
CASE_SCHEMA = "transcribe-audio.plan0065-d2-context-case.v1"
RECEIPT_SCHEMA = "transcribe-audio.plan0065-d2-receipt.v1"

EFFECT_COUNTS = {
    key: value for key, value in d0.ACTION_COUNTS.items() if key != "model_turns"
}


class Plan0065D2Error(ValueError):
    """Raised when D2 authority, normalization, or request bounds fail."""


def _hash(value: Any) -> str:
    return canonical_artifact_hash(value)


def _content(value: Mapping[str, Any]) -> dict[str, Any]:
    core = dict(value)
    core.pop("content_sha256", None)
    return {**core, "content_sha256": _hash(core)}


def _validate(value: Mapping[str, Any], label: str) -> None:
    core = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != _hash(core):
        raise Plan0065D2Error(f"{label} content hash drifted.")


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], check=False, capture_output=True, text=True
    )
    if result.returncode:
        raise Plan0065D2Error(result.stderr.strip() or "Git authority read failed.")
    return result.stdout.strip()


def repository_authority() -> dict[str, Any]:
    module = Path(__file__).resolve()
    root = Path(_git("rev-parse", "--show-toplevel")).resolve()
    relative = module.relative_to(root).as_posix()
    commit = _git("log", "-1", "--format=%H", "--", relative)
    committed = subprocess.run(
        ["git", "show", f"{commit}:{relative}"], check=False, capture_output=True
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
    if not authority["module_blob_matches"] or not authority["clean"] or authority[
        "upstream_ahead"
    ] or authority["upstream_behind"]:
        raise Plan0065D2Error("D2 repository authority is not clean and upstream-even.")
    return authority


def neutralize_uncited_factors(
    readout: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Drop only zero-point uncited factors; never invent or broaden citations."""

    value = json.loads(json.dumps(readout))
    lists: list[tuple[str, list[dict[str, Any]]]] = []
    calendar = value.get("calendar_association")
    if isinstance(calendar, dict) and isinstance(calendar.get("factors"), list):
        lists.append(("calendar_association", calendar["factors"]))
    for index, link in enumerate(value.get("person_links") or []):
        if isinstance(link, dict) and isinstance(link.get("factors"), list):
            lists.append((f"person_links[{index}]", link["factors"]))
    for index, assignment in enumerate(value.get("speaker_assignments") or []):
        if isinstance(assignment, dict) and isinstance(assignment.get("factors"), list):
            lists.append((f"speaker_assignments[{index}]", assignment["factors"]))
    neutralized = []
    for scope, factors in lists:
        kept = []
        for index, factor in enumerate(factors):
            if not isinstance(factor, dict):
                kept.append(factor)
                continue
            cited = [str(item).strip() for item in factor.get("evidence_ids") or [] if str(item).strip()]
            if cited:
                kept.append(factor)
                continue
            if str(factor.get("direction") or "").strip().lower() != "neutral":
                raise Plan0065D2Error("Uncited non-neutral factor cannot be repaired.")
            neutralized.append(
                {
                    "scope": scope,
                    "factor_index": index,
                    "factor": str(factor.get("factor") or ""),
                    "reason_code": "uncited_neutral_factor_removed",
                }
            )
        factors[:] = kept
    return value, {
        "neutralized_factor_count": len(neutralized),
        "neutralized_factors": neutralized,
        "invented_citation_count": 0,
        "changed_non_reference_field_count": 0,
    }


def build_exhaustive_clue_only_discovery(
    packet: Mapping[str, Any]
) -> dict[str, Any]:
    speakers = []
    for item in packet.get("speakers") or []:
        if not isinstance(item, Mapping):
            continue
        speakers.append(
            {
                "speaker_label": str(item.get("speaker_label") or ""),
                "transcript_clue_ids": [
                    str(clue.get("utterance_id") or "")
                    for clue in item.get("utterance_clues") or []
                    if isinstance(clue, Mapping) and clue.get("utterance_id")
                ],
                "calendar_clue_ids": [],
                "observations": [],
                "person_hints": [],
                "retrieval_terms": [],
            }
        )
    return {
        "schema_version": CLUE_DISCOVERY_READOUT_SCHEMA_VERSION,
        "speaker_clues": speakers,
        "conversation_clues": [],
        "speaker_group_hints": [],
        "mixed_speaker_hints": [],
        "warnings": ["host_exhaustive_clue_only_discovery"],
        "policy": {
            "identify_people_in_this_pass": False,
            "uses_operator_notes": False,
            "uses_gold": False,
        },
    }


def evaluate_context_gate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    correct = sum(
        item.get("disposition") == "candidate" and item.get("gold") == "correct"
        for item in rows
    )
    wrong = sum(
        item.get("disposition") == "candidate" and item.get("gold") == "wrong"
        for item in rows
    )
    incomplete = sum(
        item.get("disposition") == "candidate"
        and item.get("candidate_provenance_complete") is not True
        for item in rows
    )
    dispositions = {"candidate", "review", "abstain", "unavailable"}
    terminal = sum(item.get("disposition") in dispositions for item in rows)
    unavailable_identity = sum(
        item.get("disposition") == "unavailable" and item.get("has_candidate_identity")
        for item in rows
    )
    passed = (
        len(rows) == 39
        and terminal == 39
        and correct >= 1
        and wrong == 0
        and incomplete == 0
        and unavailable_identity == 0
    )
    return {
        "speaker_slot_count": len(rows),
        "terminal_disposition_count": terminal,
        "correct_prepared_candidate_count": correct,
        "wrong_prepared_candidate_count": wrong,
        "incomplete_candidate_provenance_count": incomplete,
        "unavailable_identity_inference_count": unavailable_identity,
        "schema_or_citation_violation_count": 0,
        "passed": passed,
        "terminal_status": "d2_pass" if passed else "context_recovery_failed",
    }


def _activation_paths(root: Path, sha: str) -> dict[str, Path]:
    base = root.expanduser().absolute()
    run = base / f"d2-activation-{sha[:24]}"
    return {"root": base, "run": run, "manifest": run / "manifest.json", "receipt": run / "receipt.json"}


def _execution_paths(root: Path, sha: str) -> dict[str, Path]:
    base = root.expanduser().absolute()
    run = base / f"d2-execution-{sha[:24]}"
    return {"root": base, "run": run, "cases": run / "cases", "receipt": run / "receipt.json"}


def _plan64_cases() -> list[dict[str, Any]]:
    paths = d0._plan0064_paths(d0.DEFAULT_PLAN0064_ROOT)
    return [read_private_object(path) for path in sorted((paths["p2"] / "cases").glob("*.json"))]


def freeze_activation(
    *, runtime_root: Path = DEFAULT_RUNTIME_ROOT, api_base_url: str = DEFAULT_API_BASE_URL
) -> dict[str, Any]:
    repository = repository_authority()
    d1_paths = d1._paths(runtime_root, D1_POLICY_SHA256)
    d1_receipt = read_private_object(d1_paths["receipt"])
    _validate(d1_receipt, "D1 receipt")
    if d1_receipt.get("content_sha256") != D1_RECEIPT_SHA256 or not d1_receipt.get(
        "development_gate", {}
    ).get("passed"):
        raise Plan0065D2Error("D1 did not authorize D2.")
    readiness = d0.provider_readiness(base_url=api_base_url)
    cases = _plan64_cases()
    partitions = dict(sorted(Counter(case["status"] if case["status"] == "context_workflow_complete" else case.get("failure_stage") for case in cases).items()))
    request_ids = sorted(
        case["document_id"]
        for case in cases
        if case.get("failure_stage") == "provider_routes_unavailable"
    )
    if partitions != {
        "context_workflow_complete": 4,
        "identity_evaluation_validation": 4,
        "provider_routes_unavailable": 4,
    } or len(request_ids) != 4:
        raise Plan0065D2Error("Plan 0064 context denominator drifted.")
    manifest = _content(
        {
            "schema_version": ACTIVATION_SCHEMA,
            "status": "d2_bounded_context_authority_frozen",
            "repository_authority": repository,
            "d0_manifest_content_sha256": d0.D0_MANIFEST_SHA256 if hasattr(d0, "D0_MANIFEST_SHA256") else d1.D0_MANIFEST_SHA256,
            "d1_policy_content_sha256": D1_POLICY_SHA256,
            "d1_receipt_content_sha256": D1_RECEIPT_SHA256,
            "plan0064_p2_receipt_content_sha256": d0.P2_RECEIPT_SHA256,
            "case_partitions": partitions,
            "primary_request_document_ids": request_ids,
            "historical_repair_count": 4,
            "historical_complete_count": 4,
            "provider_readiness": readiness,
            "request_budget": {"primary_per_case": 1, "fallback_per_case": 1},
            "fallback_policy": "skip_when_not_ready",
            "neutralization_policy": "drop_only_uncited_neutral_factors",
            "effect_counts": dict(EFFECT_COUNTS),
        }
    )
    paths = _activation_paths(runtime_root, manifest["content_sha256"])
    if paths["receipt"].exists():
        receipt = read_private_object(paths["receipt"])
        return {**receipt, "activation_manifest_path": str(paths["manifest"]), "idempotent_replay": True}
    ensure_private_tree(paths["root"], paths["run"])
    write_immutable_private_json(paths["manifest"], manifest)
    receipt = _content(
        {
            "schema_version": ACTIVATION_RECEIPT_SCHEMA,
            "status": "d2_activation_frozen_zero_effect",
            "activation_content_sha256": manifest["content_sha256"],
            "activation_file_sha256": sha256_file(paths["manifest"]),
            "primary_request_case_count": 4,
            "provider_status": readiness["status"],
            "effect_counts": dict(EFFECT_COUNTS),
        }
    )
    write_immutable_private_json(paths["receipt"], receipt)
    return {**receipt, "activation_manifest_path": str(paths["manifest"]), "idempotent_replay": False}


def _run_readout(state_root: Path, run_id: str) -> dict[str, Any]:
    run = json.loads((state_root / "app-intelligence-runs" / run_id / "run.json").read_text())
    path = Path(run["latest_model_turn_status"]["artifact_path"])
    status = json.loads(path.read_text())
    return app_intelligence_ledger.extract_json_object(str(status.get("output_text") or ""))


def _packet(state_root: Path, run_id: str) -> dict[str, Any]:
    return json.loads(
        (
            state_root
            / "app-intelligence-runs"
            / run_id
            / "artifacts/speaker-preprocessing/identity_evaluation.input.json"
        ).read_text()
    )


def _prediction(packet: Mapping[str, Any], validated: Mapping[str, Any]) -> dict[str, Any]:
    readout = validated["readout"]
    return {
        "evaluation_id": packet["evaluation_id"],
        "status": "awaiting_human_confirmation",
        "proposals": list(readout.get("speaker_assignments") or []),
        "warnings": list(readout.get("warnings") or []),
        "source_failures": list((packet.get("retrieval") or {}).get("source_failures") or []),
        "calendar_association": dict(readout.get("calendar_association") or {}),
    }


def _repaired_case(
    case: Mapping[str, Any], *, state_root: Path, canonical_people: set[str]
) -> dict[str, Any]:
    refs = case["run_references"]
    packet = _packet(state_root, refs["identity_evaluation_run_id"])
    raw = _run_readout(state_root, refs["identity_evaluation_repair_run_id"])
    normalized, audit = neutralize_uncited_factors(raw)
    validated = validate_and_score_identity_evaluation(packet, normalized)
    successor = p2._successful_case(
        document_id=case["document_id"],
        speaker_labels=[item["speaker_label"] for item in case["speaker_slots"]],
        result={"prediction": _prediction(packet, validated), "run_references": refs},
        canonical_people=canonical_people,
    )
    return _content({"schema_version": CASE_SCHEMA, "origin": "historical_reference_repair", "neutralization_audit": audit, **{k: v for k, v in successor.items() if k not in {"schema_version", "content_sha256", "action_counts"}}, "effect_counts": dict(EFFECT_COUNTS)})


def _provider_case(
    case: Mapping[str, Any], *, recording: Mapping[str, Any], state_root: Path,
    canonical_people: set[str], runner: LocalSpeakerCaseRunner,
) -> tuple[dict[str, Any], int]:
    transcript = json.loads(Path(recording["transcript_artifact"]["path"]).read_text())
    clue_packet = build_clue_discovery_packet(transcript=transcript)
    discovery = build_exhaustive_clue_only_discovery(clue_packet)
    validate_clue_discovery_readout(clue_packet, discovery)
    try:
        prepared = runner._post(
            f"/api/conversations/{case['document_id']}/speaker-preprocessing/prepare-evaluation",
            {"discovery_readout": discovery},
        )
        status = runner._execute_prepared(prepared)
        raw = runner._captured_json(status)
        packet = _packet(state_root, prepared["run_id"])
        normalized, audit = neutralize_uncited_factors(raw)
        validated = validate_and_score_identity_evaluation(packet, normalized)
        successor = p2._successful_case(
            document_id=case["document_id"],
            speaker_labels=[item["speaker_label"] for item in case["speaker_slots"]],
            result={"prediction": _prediction(packet, validated), "run_references": {"identity_evaluation_run_id": prepared["run_id"]}},
            canonical_people=canonical_people,
        )
        return _content({"schema_version": CASE_SCHEMA, "origin": "one_primary_current_route", "neutralization_audit": audit, **{k: v for k, v in successor.items() if k not in {"schema_version", "content_sha256", "action_counts"}}, "effect_counts": dict(EFFECT_COUNTS)}), 1
    except Exception as exc:
        failure = p2._failure_case(
            document_id=case["document_id"],
            speaker_labels=[item["speaker_label"] for item in case["speaker_slots"]],
            stage="plan0065_primary_route_or_validation",
            message=str(exc),
            run_references={},
        )
        return _content({"schema_version": CASE_SCHEMA, "origin": "one_primary_current_route_failed", **{k: v for k, v in failure.items() if k not in {"schema_version", "content_sha256", "action_counts"}}, "effect_counts": dict(EFFECT_COUNTS)}), 1


def execute_d2(
    activation_content_sha256: str, *, runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    state_root: Path = DEFAULT_STATE_ROOT, api_base_url: str = DEFAULT_API_BASE_URL,
) -> dict[str, Any]:
    repository_authority()
    activation_paths = _activation_paths(runtime_root, activation_content_sha256)
    for path in (activation_paths["manifest"], activation_paths["receipt"]):
        require_private_file(path, activation_paths["root"])
    activation = read_private_object(activation_paths["manifest"])
    _validate(activation, "D2 activation")
    paths = _execution_paths(runtime_root, activation_content_sha256)
    if paths["receipt"].exists():
        return replay_d2(activation_content_sha256, runtime_root=runtime_root)
    ensure_private_tree(paths["root"], paths["cases"])
    plan64 = _plan64_cases()
    p0_manifest = read_private_object(d0._plan0064_paths(d0.DEFAULT_PLAN0064_ROOT)["p0"] / "private-manifest.json")
    gold = read_private_object(d0._plan0064_paths(d0.DEFAULT_PLAN0064_ROOT)["measurement"] / "human-gold.json")
    gold_by_ref = {item["speaker_ref"]: item for item in gold["decisions"]}
    selected = {item["document_id"]: item for item in p0_manifest["evaluation_cohort"]["considered"] if item.get("disposition") == "selected_evaluation_candidate"}
    canonical = {item["canonical_person_id"] for item in p0_manifest["profile_inventory"]["active_profiles"] if item.get("identity_candidate_eligible")}
    runner = LocalSpeakerCaseRunner(base_url=api_base_url, timeout_seconds=600)
    cases = []
    primary_count = 0
    for old in plan64:
        path = paths["cases"] / f"{old['document_id']}.json"
        if path.exists():
            case = read_private_object(path)
        elif old["status"] == "context_workflow_complete":
            case = _content({"schema_version": CASE_SCHEMA, "origin": "historical_validated_complete", **{k: v for k, v in old.items() if k not in {"schema_version", "content_sha256", "action_counts"}}, "effect_counts": dict(EFFECT_COUNTS)})
            write_immutable_private_json(path, case)
        elif old.get("failure_stage") == "identity_evaluation_validation":
            case = _repaired_case(old, state_root=state_root.expanduser(), canonical_people=canonical)
            write_immutable_private_json(path, case)
        else:
            case, used = _provider_case(old, recording=selected[old["document_id"]], state_root=state_root.expanduser(), canonical_people=canonical, runner=runner)
            primary_count += used
            write_immutable_private_json(path, case)
        cases.append(case)
    rows = []
    for case in cases:
        for slot in case["speaker_slots"]:
            candidate = slot.get("candidate_person_id")
            proposal = next((item for item in slot.get("candidates") or [] if item.get("person_id") == candidate), {})
            provenance_complete = not candidate or bool(proposal.get("transcript_clue_ids")) and bool(proposal.get("provenance_source_ids"))
            gold_row = gold_by_ref[slot["speaker_ref"]]
            outcome = "correct" if candidate and gold_row["decision"] == "canonical_person" and gold_row["person_id"] == candidate else "wrong" if candidate else "other"
            disposition = slot.get("disposition") or ("unavailable" if case["status"] != "context_workflow_complete" else "abstain")
            rows.append({"speaker_ref": slot["speaker_ref"], "disposition": disposition, "gold": outcome, "candidate_provenance_complete": provenance_complete, "has_candidate_identity": bool(candidate)})
    gate = evaluate_context_gate(rows)
    receipt = _content({
        "schema_version": RECEIPT_SCHEMA,
        "status": "d2_pass_zero_effect" if gate["passed"] else "context_recovery_failed_zero_effect",
        "activation_content_sha256": activation_content_sha256,
        "case_content_sha256s": [case["content_sha256"] for case in cases],
        "case_status_counts": dict(sorted(Counter(case["status"] for case in cases).items())),
        "context_gate": gate,
        "execution_counts": {"primary_model_turn_count": primary_count, "fallback_model_turn_count": 0},
        "effect_counts": dict(EFFECT_COUNTS),
    })
    write_immutable_private_json(paths["receipt"], receipt)
    return {**receipt, "private_receipt_path": str(paths["receipt"]), "idempotent_replay": False}


def replay_d2(
    activation_content_sha256: str, *, runtime_root: Path = DEFAULT_RUNTIME_ROOT
) -> dict[str, Any]:
    paths = _execution_paths(runtime_root, activation_content_sha256)
    require_private_file(paths["receipt"], paths["root"])
    receipt = read_private_object(paths["receipt"])
    _validate(receipt, "D2 receipt")
    cases = [read_private_object(path) for path in sorted(paths["cases"].glob("*.json"))]
    if len(cases) != 12 or [case["content_sha256"] for case in cases] != receipt["case_content_sha256s"]:
        raise Plan0065D2Error("D2 case set drifted.")
    return {**receipt, "private_receipt_path": str(paths["receipt"]), "idempotent_replay": True}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("activate", "execute", "replay"))
    parser.add_argument("--activation-content-sha256")
    args = parser.parse_args()
    if args.mode == "activate":
        result = freeze_activation()
    elif args.mode == "execute":
        if not args.activation_content_sha256:
            raise SystemExit("execute requires --activation-content-sha256")
        result = execute_d2(args.activation_content_sha256)
    else:
        if not args.activation_content_sha256:
            raise SystemExit("replay requires --activation-content-sha256")
        result = replay_d2(args.activation_content_sha256)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
