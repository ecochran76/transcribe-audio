"""Freeze source-bound enrollment feasibility for reviewed Plan 0063 people."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping

from acoustic_audio_derivatives import (
    canonical_artifact_hash,
    ensure_private_tree,
    read_private_object,
    require_private_file,
    resolve_derivative_lineage_receipt,
    sha256_file,
    write_immutable_private_json,
)
from speaker_identity_evidence_execution import _source_case
import speaker_identity_plan0063_reconciliation as reconciliation


MANIFEST_SCHEMA = "transcribe-audio.plan0063-enrollment-feasibility.v1"
RECEIPT_SCHEMA = "transcribe-audio.plan0063-enrollment-feasibility-receipt.v1"
RECONCILIATION_SHA256 = (
    "82a6834165b20e9457536fbbe67e1540a583ee6dd72374296de55e5b6ccf7f05"
)
DEFAULT_RUNTIME_ROOT = Path.home() / ".local/state/transcribe-audio/plan-0063"
DEFAULT_P1_ROOT = DEFAULT_RUNTIME_ROOT / "p3-audio-lineage"
DEFAULT_CONTEXT_DATABASE = (
    Path.home()
    / ".local/state/transcribe-audio/plan-0060"
    / "p2b-context-08afc1b021a30f2a06f6e45b"
    / "context-shadow/transcripts.sqlite3"
)
DOCUMENT_ORDER = (
    "8232481d6076282d7a8e",
    "47ea79857aa1ac2d1d79",
    "92d2cd3ed6fc6c1275ca",
)
P1_AUTHORITIES = {
    "8232481d6076282d7a8e": {
        "run_id": "audio-run-abcd26aec30e1ffd488a47f5",
        "replay_receipt_sha256": (
            "9e9fd5f6a69a3f85edd0d1c261f6a3b5ee15d92ee92ba7346855ccd11cba48e4"
        ),
    },
    "47ea79857aa1ac2d1d79": {
        "run_id": "audio-run-a9581943ae2b4624c9b975a3",
        "replay_receipt_sha256": (
            "1e522dc6b0238822346738581f51ce2014569d92b210b062997918b14707e741"
        ),
    },
    "92d2cd3ed6fc6c1275ca": {
        "run_id": "audio-run-2acacf88a6c3e333e3d3d8e3",
        "replay_receipt_sha256": (
            "4a6a65c97f1618aea98e8465379a3309eb81fa38151ef2fd92e22248a150aad6"
        ),
    },
}
NEGATIVE_ACTIONS = dict(reconciliation.NEGATIVE_ACTIONS)


class Plan0063FeasibilityError(ValueError):
    """Raised when enrollment-source feasibility is incomplete or drifts."""


def _fail(message: str) -> None:
    raise Plan0063FeasibilityError(message)


def _git(repo_root: Path, arguments: list[str]) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        _fail("Repository authority could not be read.")
    return result.stdout.strip()


def _repository_authority() -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    if _git(root, ["status", "--porcelain=v1", "--untracked-files=normal"]):
        _fail("Repository must be clean before feasibility freeze.")
    if _git(
        root, ["rev-list", "--left-right", "--count", "HEAD...@{upstream}"]
    ).split() != ["0", "0"]:
        _fail("Repository must be upstream-even before feasibility freeze.")
    modules = (
        Path(__file__).resolve(),
        root / "speaker_identity_plan0063_reconciliation.py",
    )
    return {
        "commit": _git(root, ["rev-parse", "HEAD"]),
        "upstream": _git(root, ["rev-parse", "@{upstream}"]),
        "modules": {path.name: sha256_file(path) for path in modules},
    }


def _overlaps(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return (
        left["recording_id"] == right["recording_id"]
        and float(left["start_seconds"]) < float(right["end_seconds"])
        and float(right["start_seconds"]) < float(left["end_seconds"])
    )


def select_nonoverlapping_windows(
    candidates_by_person: Mapping[str, Iterable[Mapping[str, Any]]],
    *,
    maximum_per_person: int = 6,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Select longest bounded windows while preventing cross-person overlap."""

    selected: dict[str, list[dict[str, Any]]] = {}
    conflicts: list[dict[str, Any]] = []
    occupied: list[tuple[str, dict[str, Any]]] = []
    for person_id, raw_candidates in candidates_by_person.items():
        candidates = [dict(item) for item in raw_candidates]
        candidates.sort(
            key=lambda item: (
                -float(item["end_seconds"]) + float(item["start_seconds"]),
                str(item["recording_id"]),
                float(item["start_seconds"]),
            )
        )
        accepted: list[dict[str, Any]] = []
        for candidate in candidates:
            conflict_person = next(
                (
                    other_person
                    for other_person, other in occupied
                    if other_person != person_id and _overlaps(candidate, other)
                ),
                "",
            )
            if conflict_person:
                conflicts.append(
                    {
                        "proposed_person_id": person_id,
                        "conflicting_person_id": conflict_person,
                        "recording_id": candidate["recording_id"],
                        "start_seconds": candidate["start_seconds"],
                        "end_seconds": candidate["end_seconds"],
                    }
                )
                continue
            accepted.append(candidate)
            occupied.append((person_id, candidate))
            if len(accepted) >= maximum_per_person:
                break
        selected[person_id] = sorted(
            accepted,
            key=lambda item: (
                str(item["recording_id"]),
                float(item["start_seconds"]),
            ),
        )
    return selected, conflicts


def _slot_candidates(
    slot_id: str,
    case: Mapping[str, Any],
    lineage: Mapping[str, Any],
) -> list[dict[str, Any]]:
    document_id, speaker_ref = slot_id.split("::", 1)
    if document_id != case["document_id"] or speaker_ref not in case["speaker_refs"]:
        _fail("A reconciliation slot lacks an exact source speaker binding.")
    label_by_ref = dict(zip(case["speaker_refs"], case["speaker_labels"], strict=True))
    source_label = label_by_ref[speaker_ref]
    candidates: list[dict[str, Any]] = []
    for ordinal, item in enumerate(case["timeline"], start=1):
        if item.get("speaker") != source_label:
            continue
        start = round(float(item["start"]), 3)
        end = round(min(float(item["end"]), start + 15.0), 3)
        if end - start < 3.0:
            continue
        reference_id = "review-window-" + canonical_artifact_hash(
            {
                "slot_id": slot_id,
                "ordinal": ordinal,
                "start": start,
                "end": end,
                "source_sha256": lineage["source_sha256"],
            }
        )[:24]
        candidates.append(
            {
                "reference_id": reference_id,
                "slot_id": slot_id,
                "source_blob_id": lineage["source_blob_id"],
                "recording_id": case["recording_id"],
                "conversation_id": case["conversation_id"],
                "speaker_label_id": speaker_ref,
                "session_id": case["recording_id"],
                "source_sha256": lineage["source_sha256"],
                "start_seconds": start,
                "end_seconds": end,
                "source_duration_seconds": lineage["source_duration_seconds"],
                "quality_evidence": {
                    "evidence_id": "quality-"
                    + str(lineage["audio_quality_sha256"])[:24],
                    "sha256": lineage["audio_quality_sha256"],
                },
                "device_class": "unverified-mobile-recorder",
                "acoustic_conditions": [
                    "conversation_recording",
                    "diarized_speech_window",
                ],
                "lineage": dict(lineage),
                "data_split": "development_training_candidate",
                "future_holdout_excluded": True,
            }
        )
    return candidates


def build_feasibility_manifest(
    reconciliation_manifest: Mapping[str, Any],
    *,
    cases: Mapping[str, Mapping[str, Any]],
    lineages: Mapping[str, Mapping[str, Any]],
    repository_authority: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a no-profile source-window review authority."""

    if (
        reconciliation_manifest.get("content_sha256") != RECONCILIATION_SHA256
        or reconciliation_manifest.get("status")
        != "pending_human_grouping_and_binding_review"
        or any((reconciliation_manifest.get("negative_actions") or {}).values())
    ):
        _fail("The Plan 0063 reconciliation source is invalid.")
    enrollment_candidates = reconciliation_manifest.get("enrollment_candidates")
    if not isinstance(enrollment_candidates, list):
        _fail("The reconciliation enrollment candidates are unavailable.")
    candidates_by_person: dict[str, list[dict[str, Any]]] = {}
    for candidate in enrollment_candidates:
        if not isinstance(candidate, Mapping):
            _fail("An enrollment candidate is invalid.")
        person_id = str(candidate.get("proposed_person_id") or "")
        windows: list[dict[str, Any]] = []
        for slot_id in candidate.get("member_slot_ids") or []:
            document_id = str(slot_id).split("::", 1)[0]
            case = cases.get(document_id)
            lineage = lineages.get(document_id)
            if not case or not lineage:
                _fail("An enrollment candidate lacks source or P1 lineage.")
            windows.extend(_slot_candidates(str(slot_id), case, lineage))
        candidates_by_person[person_id] = windows
    selected, conflicts = select_nonoverlapping_windows(candidates_by_person)
    proposals = []
    for candidate in enrollment_candidates:
        person_id = str(candidate["proposed_person_id"])
        windows = selected[person_id]
        total_seconds = round(
            sum(float(item["end_seconds"]) - float(item["start_seconds"]) for item in windows),
            3,
        )
        feasible = len(windows) >= 2 and total_seconds >= 8.0
        proposals.append(
            {
                "proposed_person_id": person_id,
                "member_slot_ids": list(candidate.get("member_slot_ids") or []),
                "status": (
                    "source_feasible_pending_human_review"
                    if feasible
                    else "ineligible_insufficient_nonoverlapping_speech"
                ),
                "window_count": len(windows),
                "usable_seconds": total_seconds,
                "source_windows": windows,
                "device_metadata_status": "unverified",
                "enrollment_authorized": False,
            }
        )
    exclusion_rows = sorted(
        {
            (item["recording_id"], item["source_sha256"])
            for proposal in proposals
            for item in proposal["source_windows"]
        }
    )
    metrics = {
        "person_candidate_count": len(proposals),
        "source_feasible_count": sum(
            item["status"] == "source_feasible_pending_human_review"
            for item in proposals
        ),
        "ineligible_count": sum(item["status"].startswith("ineligible_") for item in proposals),
        "source_window_count": sum(item["window_count"] for item in proposals),
        "source_recording_count": len(exclusion_rows),
        "source_conflict_count": len(conflicts),
        "future_holdout_exclusion_count": len(exclusion_rows),
    }
    core = {
        "schema_version": MANIFEST_SCHEMA,
        "status": "source_feasibility_ready_pending_human_review",
        "reconciliation_content_sha256": RECONCILIATION_SHA256,
        "repository_authority": dict(repository_authority or {}),
        "metrics": metrics,
        "person_source_proposals": proposals,
        "source_conflicts": conflicts,
        "future_holdout_exclusions": [
            {"recording_id": recording_id, "source_sha256": source_sha256}
            for recording_id, source_sha256 in exclusion_rows
        ],
        "live_mutation_count": 0,
        "negative_actions": dict(NEGATIVE_ACTIONS),
    }
    return {**core, "content_sha256": canonical_artifact_hash(core)}


def _paths(runtime_root: Path, content_sha256: str) -> dict[str, Path]:
    root = runtime_root.expanduser().absolute()
    run = root / f"p3-enrollment-feasibility-{content_sha256[:24]}"
    return {
        "root": root,
        "run": run,
        "manifest": run / "private-manifest.json",
        "receipt": run / "receipt.json",
    }


def freeze_exact_feasibility(
    *,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    p1_root: Path = DEFAULT_P1_ROOT,
    context_database: Path = DEFAULT_CONTEXT_DATABASE,
) -> dict[str, Any]:
    """Freeze exact P3 source feasibility after replaying P1 and P2 authority."""

    replay = reconciliation.replay_reconciliation(
        content_sha256=RECONCILIATION_SHA256, runtime_root=runtime_root
    )
    reconciled = read_private_object(Path(replay["manifest_path"]))
    cases = {document_id: _source_case(context_database, document_id) for document_id in DOCUMENT_ORDER}
    lineages = {
        document_id: resolve_derivative_lineage_receipt(
            authority["run_id"],
            runtime_root=p1_root,
            replay_receipt_sha256=authority["replay_receipt_sha256"],
        )
        for document_id, authority in P1_AUTHORITIES.items()
    }
    manifest = build_feasibility_manifest(
        reconciled,
        cases=cases,
        lineages=lineages,
        repository_authority=_repository_authority(),
    )
    if manifest["metrics"]["person_candidate_count"] != 5:
        _fail("The exact enrollment candidate denominator drifted.")
    paths = _paths(runtime_root, manifest["content_sha256"])
    if paths["receipt"].exists():
        return replay_feasibility(
            content_sha256=manifest["content_sha256"], runtime_root=runtime_root
        )
    if paths["run"].exists():
        _fail("A partial feasibility directory already exists.")
    ensure_private_tree(paths["root"], paths["run"])
    write_immutable_private_json(paths["manifest"], manifest)
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "source_feasibility_frozen_pending_human_review",
        "content_sha256": manifest["content_sha256"],
        "manifest_sha256": sha256_file(paths["manifest"]),
        "metrics": manifest["metrics"],
        "live_mutation_count": 0,
        "negative_actions_preserved": True,
    }
    write_immutable_private_json(paths["receipt"], receipt)
    return {
        **receipt,
        "manifest_path": str(paths["manifest"]),
        "receipt_path": str(paths["receipt"]),
        "idempotent_replay": False,
    }


def replay_feasibility(
    *, content_sha256: str, runtime_root: Path = DEFAULT_RUNTIME_ROOT
) -> dict[str, Any]:
    paths = _paths(runtime_root, content_sha256)
    require_private_file(paths["manifest"], paths["root"])
    require_private_file(paths["receipt"], paths["root"])
    manifest = read_private_object(paths["manifest"])
    receipt = read_private_object(paths["receipt"])
    core = {key: value for key, value in manifest.items() if key != "content_sha256"}
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA
        or manifest.get("content_sha256") != canonical_artifact_hash(core)
        or manifest.get("content_sha256") != content_sha256
        or any((manifest.get("negative_actions") or {}).values())
        or receipt.get("schema_version") != RECEIPT_SCHEMA
        or receipt.get("content_sha256") != content_sha256
        or receipt.get("manifest_sha256") != sha256_file(paths["manifest"])
        or receipt.get("metrics") != manifest.get("metrics")
        or receipt.get("live_mutation_count") != 0
        or receipt.get("negative_actions_preserved") is not True
    ):
        _fail("The frozen enrollment feasibility evidence drifted.")
    return {
        **receipt,
        "manifest_path": str(paths["manifest"]),
        "receipt_path": str(paths["receipt"]),
        "idempotent_replay": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "replay"))
    parser.add_argument("--content-sha256", default="")
    args = parser.parse_args()
    result = (
        freeze_exact_feasibility()
        if args.command == "freeze"
        else replay_feasibility(content_sha256=args.content_sha256)
    )
    print(
        {
            "status": result["status"],
            "content_sha256": result["content_sha256"],
            "manifest_sha256": result["manifest_sha256"],
            "metrics": result["metrics"],
            "live_mutation_count": result["live_mutation_count"],
            "idempotent_replay": result["idempotent_replay"],
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
