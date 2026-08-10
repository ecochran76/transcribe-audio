"""Plan 0064 P3 deterministic conversation-level speaker resolver."""

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
from speaker_identity_plan0064_p2 import replay_p2


PREVIEW_SCHEMA = "transcribe-audio.plan0064-p3-preview.v1"
RESOLUTION_SCHEMA = "transcribe-audio.plan0064-p3-resolution.v1"
RECEIPT_SCHEMA = "transcribe-audio.plan0064-p3-receipt.v1"
POLICY_VERSION = "plan0064-conversation-resolver-v1"
ACTION_COUNTS = {
    "speaker_assignments": 0,
    "new_enrollments": 0,
    "profile_mutations": 0,
    "knowledge_writes": 0,
    "provider_writes": 0,
    "external_provider_writes": 0,
}


class Plan0064P3Error(ValueError):
    """Raised when a frozen pillar or conversation resolution drifts."""


def _hash(value: Any) -> str:
    return canonical_artifact_hash(value)


def _content_addressed(value: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(value)
    body.pop("content_sha256", None)
    return {**body, "content_sha256": _hash(body)}


def _read(path: Path) -> dict[str, Any]:
    value = read_private_object(path)
    if not isinstance(value, dict):
        raise Plan0064P3Error(f"Private artifact is not an object: {path}")
    return value


def _load_frozen_p1(
    p0_content_sha256: str, *, runtime_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the unique immutable P1 receipt without reopening mutable P0 inputs."""
    matches = []
    for root in sorted(runtime_root.expanduser().absolute().glob("p1-*")):
        evidence_path = root / "private-acoustic-evidence.json"
        receipt_path = root / "receipt.json"
        if not evidence_path.is_file() or not receipt_path.is_file():
            continue
        require_private_file(evidence_path, root)
        require_private_file(receipt_path, root)
        evidence, receipt = _read(evidence_path), _read(receipt_path)
        if receipt.get("p0_content_sha256") != p0_content_sha256:
            continue
        if (
            receipt.get("content_sha256")
            != _hash({key: value for key, value in receipt.items() if key != "content_sha256"})
            or evidence.get("content_sha256")
            != _hash({key: value for key, value in evidence.items() if key != "content_sha256"})
            or receipt.get("evidence_content_sha256") != evidence["content_sha256"]
            or receipt.get("evidence_file_sha256") != sha256_file(evidence_path)
            or evidence.get("p0_content_sha256") != p0_content_sha256
            or any(receipt.get("action_counts", {}).values())
            or any(evidence.get("action_counts", {}).values())
        ):
            raise Plan0064P3Error("A matching frozen P1 artifact failed validation.")
        matches.append((evidence, receipt))
    if len(matches) != 1:
        raise Plan0064P3Error("P3 requires exactly one matching frozen P1 artifact.")
    return matches[0]


def _acoustic_view(slot: Mapping[str, Any]) -> dict[str, Any]:
    status = str(slot.get("status") or "abstain")
    person_id = str(slot.get("candidate_person_id") or "") or None
    supporting_models = int(slot.get("supporting_model_count") or 0)
    if status == "candidate" and person_id and supporting_models >= 2:
        disposition, reason = "candidate", "multi_model_acoustic_support"
    elif status == "review" and person_id:
        disposition, reason = "review", "single_model_acoustic_support"
    else:
        disposition, reason, person_id = (
            "abstain",
            str(slot.get("reason_code") or "acoustic_unavailable"),
            None,
        )
    alternatives = sorted(
        {
            str(row.get("top_canonical_person_id") or "")
            for row in slot.get("model_rows") or []
            if isinstance(row, Mapping)
            and row.get("binding_eligible") is True
            and str(row.get("top_canonical_person_id") or "")
        }
    )
    return {
        "disposition": disposition,
        "reason_code": reason,
        "candidate_person_id": person_id,
        "alternative_person_ids": alternatives,
        "confidence_band": str(slot.get("confidence_band") or "none"),
        "supporting_model_count": supporting_models,
        "probe_sha256": str(slot.get("probe_sha256") or ""),
    }


def _context_view(slot: Mapping[str, Any]) -> dict[str, Any]:
    disposition = str(slot.get("disposition") or "unavailable")
    person_id = str(slot.get("candidate_person_id") or "") or None
    if disposition != "candidate":
        person_id = None
    candidates = [
        dict(item) for item in slot.get("candidates") or [] if isinstance(item, Mapping)
    ]
    alternatives = sorted(
        {
            str(item.get("prepared_person_id") or "")
            for item in candidates
            if str(item.get("prepared_person_id") or "")
        }
    )
    contradictions = sorted(
        {
            str(evidence_id)
            for item in candidates
            for factor in item.get("factors") or []
            if isinstance(factor, Mapping)
            and str(factor.get("direction") or "") == "contradict"
            for evidence_id in factor.get("evidence_ids") or []
            if str(evidence_id)
        }
    )
    return {
        "disposition": disposition,
        "reason_code": str(slot.get("reason_code") or "context_unavailable"),
        "candidate_person_id": person_id,
        "alternative_person_ids": alternatives,
        "contradiction_evidence_ids": contradictions,
        "candidates": candidates,
    }


def _combined_view(acoustic: Mapping[str, Any], context: Mapping[str, Any]) -> dict[str, Any]:
    acoustic_person = acoustic.get("candidate_person_id")
    context_person = context.get("candidate_person_id")
    alternatives = sorted(
        set(acoustic.get("alternative_person_ids") or [])
        | set(context.get("alternative_person_ids") or [])
        | ({str(acoustic_person)} if acoustic_person else set())
        | ({str(context_person)} if context_person else set())
    )
    if acoustic_person and context_person:
        if acoustic_person == context_person:
            disposition, reason, person_id = "candidate", "pillar_agreement", acoustic_person
        else:
            disposition, reason, person_id = "abstain", "pillar_conflict", None
    elif acoustic_person:
        disposition, reason, person_id = "review", "acoustic_only_support", None
    elif context_person:
        disposition, reason, person_id = "review", "context_only_support", None
    elif "review" in {acoustic["disposition"], context["disposition"]}:
        disposition, reason, person_id = "review", "incomplete_pillar_support", None
    elif acoustic["disposition"] == context["disposition"] == "unavailable":
        disposition, reason, person_id = "unavailable", "both_pillars_unavailable", None
    else:
        disposition, reason, person_id = "abstain", "no_joined_candidate", None
    return {
        "disposition": disposition,
        "reason_code": reason,
        "candidate_person_id": person_id,
        "alternative_person_ids": alternatives,
        "contradiction_evidence_ids": list(context.get("contradiction_evidence_ids") or []),
    }


def _shared_context_proposal(rows: Sequence[Mapping[str, Any]]) -> bool:
    proposal_sets = []
    for row in rows:
        person_id = row["combined"]["candidate_person_id"]
        proposal_sets.append(
            {
                str(item.get("proposal_id") or "")
                for item in row["context"].get("candidates") or []
                if item.get("status") == "candidate_match"
                and item.get("prepared_person_id") == person_id
                and str(item.get("proposal_id") or "")
            }
        )
    return bool(proposal_sets and set.intersection(*proposal_sets))


def _enforce_global_bindings(rows: list[dict[str, Any]]) -> None:
    by_person: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        person_id = row["combined"].get("candidate_person_id")
        if person_id:
            by_person.setdefault(str(person_id), []).append(row)
    for duplicated in by_person.values():
        if len(duplicated) < 2:
            continue
        if _shared_context_proposal(duplicated):
            for row in duplicated:
                row["combined"]["reason_code"] = "pillar_agreement_same_person_multi_label"
            continue
        for row in duplicated:
            row["combined"].update(
                {
                    "disposition": "review",
                    "reason_code": "global_person_collision",
                    "candidate_person_id": None,
                }
            )


def _residual_supported(row: Mapping[str, Any], case: Mapping[str, Any]) -> bool:
    context = row["context"]
    person_id = context.get("candidate_person_id")
    if not person_id or case.get("provider_failures"):
        return False
    prepared_people = set(context.get("alternative_person_ids") or [])
    matching = [
        item
        for item in context.get("candidates") or []
        if item.get("status") == "candidate_match"
        and item.get("prepared_person_id") == person_id
    ]
    if prepared_people != {person_id} or len(matching) != 1:
        return False
    candidate = matching[0]
    if not candidate.get("transcript_clue_ids") or not candidate.get("provenance_source_ids"):
        return False
    return not any(
        isinstance(factor, Mapping)
        and factor.get("direction") == "contradict"
        and factor.get("strength") in {"strong", "decisive"}
        for factor in candidate.get("factors") or []
    )


def resolve_conversation(
    acoustic_recording: Mapping[str, Any], context_case: Mapping[str, Any]
) -> dict[str, Any]:
    if acoustic_recording.get("document_id") != context_case.get("document_id"):
        raise Plan0064P3Error("P1 and P2 document bindings differ.")
    acoustic_slots = list(acoustic_recording.get("speaker_slots") or [])
    context_slots = list(context_case.get("speaker_slots") or [])
    if [item.get("speaker_ref") for item in acoustic_slots] != [
        item.get("speaker_ref") for item in context_slots
    ]:
        raise Plan0064P3Error("P1 and P2 speaker-slot order differs.")
    rows = []
    for acoustic_slot, context_slot in zip(acoustic_slots, context_slots):
        acoustic = _acoustic_view(acoustic_slot)
        context = _context_view(context_slot)
        combined = _combined_view(acoustic, context)
        rows.append(
            {
                "speaker_ref": acoustic_slot["speaker_ref"],
                "speaker_label": acoustic_slot["speaker_label"],
                "acoustic": acoustic,
                "context": context,
                "combined": combined,
                "residual_policy": dict(combined),
            }
        )
    _enforce_global_bindings(rows)
    for row in rows:
        row["residual_policy"] = dict(row["combined"])
    accepted = [row for row in rows if row["combined"]["disposition"] == "candidate"]
    remaining = [row for row in rows if row["combined"]["disposition"] != "candidate"]
    accepted_people = {row["combined"]["candidate_person_id"] for row in accepted}
    if (
        len(rows) == 3
        and len(accepted) == 2
        and len(accepted_people) == 2
        and len(remaining) == 1
        and _residual_supported(remaining[0], context_case)
        and remaining[0]["context"]["candidate_person_id"] not in accepted_people
    ):
        remaining[0]["residual_policy"].update(
            {
                "disposition": "candidate",
                "reason_code": "two_known_plus_one_independently_supported_residual",
                "candidate_person_id": remaining[0]["context"]["candidate_person_id"],
            }
        )
    return {
        "document_id": acoustic_recording["document_id"],
        "transcript_sha256": acoustic_recording["transcript_sha256"],
        "source_media_sha256": acoustic_recording["source_media_sha256"],
        "speaker_slots": rows,
    }


def build_p3_preview(
    p0_content_sha256: str, *, runtime_root: Path = DEFAULT_RUNTIME_ROOT
) -> dict[str, Any]:
    p1_evidence, p1_receipt = _load_frozen_p1(
        p0_content_sha256, runtime_root=runtime_root
    )
    p2_receipt = replay_p2(p0_content_sha256, runtime_root=runtime_root)
    return _content_addressed(
        {
            "schema_version": PREVIEW_SCHEMA,
            "status": "ready_for_deterministic_resolution",
            "policy_version": POLICY_VERSION,
            "p0_content_sha256": p0_content_sha256,
            "p1_evidence_content_sha256": p1_evidence["content_sha256"],
            "p1_receipt_content_sha256": p1_receipt["content_sha256"],
            "p2_receipt_content_sha256": p2_receipt["content_sha256"],
            "p2_case_set_sha256": p2_receipt["case_set_sha256"],
            "recording_count": p1_evidence["summary"]["recording_count"],
            "speaker_slot_count": p1_evidence["summary"]["speaker_slot_count"],
            "conditions": ["acoustic", "context", "combined", "residual_policy"],
            "will_read_gold": False,
            "will_apply_speaker_identity": False,
            "action_counts": dict(ACTION_COUNTS),
        }
    )


def execute_p3(
    p0_content_sha256: str, *, runtime_root: Path = DEFAULT_RUNTIME_ROOT
) -> dict[str, Any]:
    preview = build_p3_preview(p0_content_sha256, runtime_root=runtime_root)
    p1_evidence, _p1_receipt = _load_frozen_p1(
        p0_content_sha256, runtime_root=runtime_root
    )
    p2_receipt = replay_p2(p0_content_sha256, runtime_root=runtime_root)
    cases_root = Path(p2_receipt["private_cases_root"])
    context_by_document = {
        case["document_id"]: case
        for path in sorted(cases_root.glob("*.json"))
        for case in [_read(path)]
    }
    acoustic_recordings = list(p1_evidence.get("recordings") or [])
    if set(context_by_document) != {
        str(item.get("document_id") or "") for item in acoustic_recordings
    }:
        raise Plan0064P3Error("P1 and P2 recording denominators differ.")
    recordings = [
        resolve_conversation(item, context_by_document[item["document_id"]])
        for item in acoustic_recordings
    ]
    slots = [slot for recording in recordings for slot in recording["speaker_slots"]]
    summary = {
        "recording_count": len(recordings),
        "speaker_slot_count": len(slots),
        "condition_disposition_counts": {
            condition: dict(
                sorted(Counter(slot[condition]["disposition"] for slot in slots).items())
            )
            for condition in ("acoustic", "context", "combined", "residual_policy")
        },
        "combined_reason_code_counts": dict(
            sorted(Counter(slot["combined"]["reason_code"] for slot in slots).items())
        ),
        "residual_acceptance_count": sum(
            slot["residual_policy"]["reason_code"]
            == "two_known_plus_one_independently_supported_residual"
            for slot in slots
        ),
    }
    resolution = _content_addressed(
        {
            "schema_version": RESOLUTION_SCHEMA,
            "status": "complete_private_shadow_resolution",
            "preview_content_sha256": preview["content_sha256"],
            "policy_version": POLICY_VERSION,
            "recordings": recordings,
            "summary": summary,
            "contains_gold": False,
            "will_apply_speaker_identity": False,
            "action_counts": dict(ACTION_COUNTS),
        }
    )
    root = runtime_root.expanduser().absolute() / f"p3-{preview['content_sha256'][:24]}"
    resolution_path, receipt_path = root / "private-resolution.json", root / "receipt.json"
    ensure_private_tree(root, root)
    if resolution_path.exists() or receipt_path.exists():
        return replay_p3(p0_content_sha256, runtime_root=runtime_root)
    write_immutable_private_json(resolution_path, resolution)
    receipt = _content_addressed(
        {
            "schema_version": RECEIPT_SCHEMA,
            "status": "p3_complete_zero_effect",
            "preview_content_sha256": preview["content_sha256"],
            "resolution_content_sha256": resolution["content_sha256"],
            "resolution_file_sha256": sha256_file(resolution_path),
            "summary": summary,
            "action_counts": dict(ACTION_COUNTS),
        }
    )
    write_immutable_private_json(receipt_path, receipt)
    return {
        **receipt,
        "private_resolution_path": str(resolution_path),
        "private_receipt_path": str(receipt_path),
        "idempotent_replay": False,
    }


def replay_p3(
    p0_content_sha256: str, *, runtime_root: Path = DEFAULT_RUNTIME_ROOT
) -> dict[str, Any]:
    preview = build_p3_preview(p0_content_sha256, runtime_root=runtime_root)
    root = runtime_root.expanduser().absolute() / f"p3-{preview['content_sha256'][:24]}"
    resolution_path, receipt_path = root / "private-resolution.json", root / "receipt.json"
    require_private_file(resolution_path, root)
    require_private_file(receipt_path, root)
    resolution, receipt = _read(resolution_path), _read(receipt_path)
    if (
        resolution.get("preview_content_sha256") != preview["content_sha256"]
        or resolution.get("content_sha256")
        != _hash({key: value for key, value in resolution.items() if key != "content_sha256"})
        or receipt.get("resolution_content_sha256") != resolution["content_sha256"]
        or receipt.get("resolution_file_sha256") != sha256_file(resolution_path)
        or receipt.get("content_sha256")
        != _hash({key: value for key, value in receipt.items() if key != "content_sha256"})
        or receipt.get("action_counts") != ACTION_COUNTS
    ):
        raise Plan0064P3Error("The frozen P3 resolution or receipt drifted.")
    return {
        **receipt,
        "private_resolution_path": str(resolution_path),
        "private_receipt_path": str(receipt_path),
        "idempotent_replay": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preview", "execute", "replay"))
    parser.add_argument("--p0-content-sha256", required=True)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    args = parser.parse_args(argv)
    action = {
        "preview": build_p3_preview,
        "execute": execute_p3,
        "replay": replay_p3,
    }[args.action]
    result = action(args.p0_content_sha256, runtime_root=args.runtime_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
