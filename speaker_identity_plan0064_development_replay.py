"""Plan 0064 exact reviewed-development replay over the installed identity state.

This lane is development evidence, not unseen evaluation.  It scores the ten
Plan 0062 reviewed speaker clips against the current Plan 0064 profile
inventory, replays the pre-gold contextual suggestions through the accepted
Plan 0063 canonical people, and runs the Plan 0064 conversation resolver.  It
records the resulting quality gate without applying identities or writing to
providers.
"""

from __future__ import annotations

import argparse
from array import array
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import acoustic_verification as verification
from acoustic_audio_derivatives import (
    canonical_artifact_hash,
    ensure_private_tree,
    read_private_object,
    require_private_file,
    sha256_file,
    write_immutable_private_json,
)
from speaker_identity_plan0063_private_rehearsal import (
    validate_reviewed_transition,
)
from speaker_identity_plan0064_p0 import DEFAULT_RUNTIME_ROOT
from speaker_identity_plan0064_p1 import (
    ACTION_COUNTS,
    DEFAULT_PROFILE_ROOT,
    DEFAULT_REFERENCE_ROOT,
    DEFAULT_THRESHOLD_APPLICATION,
    _CachingAdapter,
    _decode,
    _score_slot,
    _thresholds,
)
from speaker_identity_plan0064_p2 import _phase_safe_p0
from speaker_identity_plan0064_p3 import resolve_conversation
from speaker_identity_plan0064_p4_measurement import (
    DEVELOPMENT_GATE_SCHEMA,
    HIGH_SUPPORT_REASONS,
)


PREVIEW_SCHEMA = "transcribe-audio.plan0064-development-replay-preview.v1"
EVIDENCE_SCHEMA = "transcribe-audio.plan0064-development-replay-evidence.v1"
RECEIPT_SCHEMA = "transcribe-audio.plan0064-development-replay-receipt.v1"
PLAN0062_P3_RECEIPT_CONTENT_SHA256 = (
    "6ee20bc30364af922063591dd53c22f8ea73da2466c182c81f5e31771864be4e"
)
PLAN0062_P3_MANIFEST_SHA256 = (
    "d0f8f8959e6d88b0ddcdc8b2dc5cf122487556ebe73a0d958a8e539f9d5bb052"
)
PLAN0062_P4_RECEIPT_CONTENT_SHA256 = (
    "bbdd481c2212401492786041ddfdb5ff1b4e7ff7774af5b33e0917d40987031d"
)
PLAN0062_P4_MANIFEST_SHA256 = (
    "420e49c92e24628643f05714e66c9713a4a8296dd523ef1b09d51105446d9bc8"
)
PLAN0063_TRANSITION_SHA256 = (
    "75166646421378e2fce4aee1e21c35a6d73fdfdbdb5b37297e4c13fc1b8663dc"
)
DEFAULT_PLAN0062_P3_ROOT = Path(
    "~/.local/state/transcribe-audio/plan-0062/"
    "p3-contextual-join-6d405b39f7f72e9ab81c155c"
)
DEFAULT_PLAN0062_P4_ROOT = Path(
    "~/.local/state/transcribe-audio/plan-0062/"
    "p4-human-review-bbdd481c2212401492786041"
)
DEFAULT_PLAN0063_TRANSITION = Path(
    "~/.local/state/transcribe-audio/plan-0063/"
    "p5-private-copy-rehearsal-75166646421378e2fce4aee1/"
    "reviewed-transition.json"
)


class Plan0064DevelopmentReplayError(ValueError):
    """Raised when reviewed-development authority or replay evidence drifts."""


def _hash(value: Any) -> str:
    return canonical_artifact_hash(value)


def _content_addressed(value: Mapping[str, Any]) -> dict[str, Any]:
    core = dict(value)
    core.pop("content_sha256", None)
    return {**core, "content_sha256": _hash(core)}


def _read(path: Path) -> dict[str, Any]:
    value = read_private_object(path)
    if not isinstance(value, dict):
        raise Plan0064DevelopmentReplayError(
            f"Private development authority is not an object: {path}"
        )
    return value


def _validate_hash(value: Mapping[str, Any], *, label: str) -> None:
    core = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != _hash(core):
        raise Plan0064DevelopmentReplayError(f"{label} content hash drifted.")


def _normalized(value: Any) -> str:
    return " ".join(str(value or "").split()).casefold()


def _repository_authority() -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=normal"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    even = subprocess.run(
        ["git", "rev-list", "--left-right", "--count", "HEAD...@{upstream}"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    module_commit = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", Path(__file__).name],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    module_commit_sha = module_commit.stdout.strip()
    committed = subprocess.run(
        ["git", "show", f"{module_commit_sha}:{Path(__file__).name}"],
        cwd=root,
        capture_output=True,
        check=False,
    )
    local_sha256 = sha256_file(Path(__file__).resolve())
    if (
        status.returncode
        or status.stdout.strip()
        or even.returncode
        or even.stdout.split() != ["0", "0"]
        or module_commit.returncode
        or len(module_commit_sha) != 40
        or committed.returncode
        or hashlib.sha256(committed.stdout).hexdigest() != local_sha256
    ):
        raise Plan0064DevelopmentReplayError(
            "Development replay requires a clean, upstream-even committed module."
    )
    return {
        "module_commit": module_commit_sha,
        "module_name": Path(__file__).name,
        "module_sha256": local_sha256,
        "clean": True,
        "upstream_ahead": 0,
        "upstream_behind": 0,
    }


def _private_sources(
    *,
    p0_content_sha256: str,
    runtime_root: Path,
    p3_root: Path,
    p4_root: Path,
    transition_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    p0_receipt, _hydration_bridge = _phase_safe_p0(
        p0_content_sha256, runtime_root=runtime_root
    )
    p0_manifest = _read(Path(str(p0_receipt["private_manifest_path"])))
    selected_p3_root = p3_root.expanduser().absolute()
    selected_p4_root = p4_root.expanduser().absolute()
    selected_transition = transition_path.expanduser().absolute()
    p3_manifest_path = selected_p3_root / "private-manifest.json"
    p3_receipt_path = selected_p3_root / "receipt.json"
    p4_manifest_path = selected_p4_root / "private-manifest.json"
    p4_receipt_path = selected_p4_root / "receipt.json"
    for path, root in (
        (p3_manifest_path, selected_p3_root),
        (p3_receipt_path, selected_p3_root),
        (p4_manifest_path, selected_p4_root),
        (p4_receipt_path, selected_p4_root),
        (selected_transition, selected_transition.parent.parent),
    ):
        require_private_file(path, root)
    p3_manifest, p3_receipt = _read(p3_manifest_path), _read(p3_receipt_path)
    p4_manifest, p4_receipt = _read(p4_manifest_path), _read(p4_receipt_path)
    transition = _read(selected_transition)
    if (
        p3_receipt.get("content_sha256")
        != PLAN0062_P3_RECEIPT_CONTENT_SHA256
        or sha256_file(p3_manifest_path) != PLAN0062_P3_MANIFEST_SHA256
        or p3_receipt.get("manifest_sha256") != PLAN0062_P3_MANIFEST_SHA256
        or p4_receipt.get("content_sha256")
        != PLAN0062_P4_RECEIPT_CONTENT_SHA256
        or sha256_file(p4_manifest_path) != PLAN0062_P4_MANIFEST_SHA256
        or p4_receipt.get("manifest_sha256") != PLAN0062_P4_MANIFEST_SHA256
        or p4_receipt.get("p3_content_sha256")
        != PLAN0062_P3_RECEIPT_CONTENT_SHA256
        or validate_reviewed_transition(transition) != PLAN0063_TRANSITION_SHA256
        or any((p3_manifest.get("negative_actions") or {}).values())
        or any((p4_manifest.get("negative_actions") or {}).values())
    ):
        raise Plan0064DevelopmentReplayError(
            "The reviewed Plan 0062/0063 development authority drifted."
        )
    packet = p4_manifest.get("packet")
    clips = p4_manifest.get("audio_clips")
    if (
        not isinstance(packet, Mapping)
        or packet.get("content_sha256")
        != PLAN0062_P4_RECEIPT_CONTENT_SHA256
        or not isinstance(clips, list)
        or len(clips) != 10
        or len(packet.get("cards") or []) != 10
        or len(p3_manifest.get("results") or []) != 3
        or len(transition.get("slot_bindings") or []) != 9
    ):
        raise Plan0064DevelopmentReplayError(
            "The reviewed development denominator is incomplete."
        )
    clip_by_slot = {
        str(item.get("slot_id") or ""): item
        for item in clips
        if isinstance(item, Mapping)
    }
    if len(clip_by_slot) != 10:
        raise Plan0064DevelopmentReplayError("Development clips repeat a speaker slot.")
    for slot_id, clip in clip_by_slot.items():
        path = selected_p4_root / "preview" / str(clip.get("relative_path") or "")
        require_private_file(path, selected_p4_root)
        if not slot_id or sha256_file(path) != clip.get("sha256"):
            raise Plan0064DevelopmentReplayError("A reviewed speaker clip drifted.")
    return p0_manifest, p3_manifest, p4_manifest, transition


def build_development_preview(
    p0_content_sha256: str,
    *,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    p3_root: Path = DEFAULT_PLAN0062_P3_ROOT,
    p4_root: Path = DEFAULT_PLAN0062_P4_ROOT,
    transition_path: Path = DEFAULT_PLAN0063_TRANSITION,
    threshold_application: Path = DEFAULT_THRESHOLD_APPLICATION,
) -> dict[str, Any]:
    p0, p3, p4, transition = _private_sources(
        p0_content_sha256=p0_content_sha256,
        runtime_root=runtime_root,
        p3_root=p3_root,
        p4_root=p4_root,
        transition_path=transition_path,
    )
    profiles = p0["profile_inventory"]["active_profiles"]
    candidate_ids = p0["profile_inventory"]["candidate_ids"]
    thresholds = _thresholds(threshold_application, candidate_ids)
    return _content_addressed(
        {
            "schema_version": PREVIEW_SCHEMA,
            "status": "ready_for_reviewed_development_replay",
            "p0_content_sha256": p0_content_sha256,
            "active_profile_set_sha256": p0["profile_inventory"][
                "active_profile_set_sha256"
            ],
            "plan0062_p3_manifest_sha256": PLAN0062_P3_MANIFEST_SHA256,
            "plan0062_p4_manifest_sha256": PLAN0062_P4_MANIFEST_SHA256,
            "plan0063_transition_sha256": transition["content_sha256"],
            "threshold_authority": thresholds,
            "recording_count": len(p3["results"]),
            "speaker_slot_count": len(p4["audio_clips"]),
            "gold_named_slot_count": len(transition["slot_bindings"]),
            "active_profile_count": len(profiles),
            "candidate_ids": list(candidate_ids),
            "repository_authority": _repository_authority(),
            "source_corpus": "plan0063_reviewed_three_conversation",
            "source_disjoint_evaluation": False,
            "will_apply_speaker_identity": False,
            "action_counts": dict(ACTION_COUNTS),
        }
    )


def _person_indexes(transition: Mapping[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    by_name: dict[str, str] = {}
    by_email: dict[str, str] = {}
    for person in transition.get("canonical_people") or []:
        person_id = str(person.get("person_id") or "")
        name = _normalized(person.get("primary_name"))
        if not person_id or not name or name in by_name:
            raise Plan0064DevelopmentReplayError(
                "Canonical development people are incomplete or ambiguous."
            )
        by_name[name] = person_id
        for identity in person.get("external_identities") or []:
            if not isinstance(identity, Mapping) or identity.get("kind") != "email":
                continue
            email = _normalized(identity.get("value"))
            if not email or (email in by_email and by_email[email] != person_id):
                raise Plan0064DevelopmentReplayError(
                    "Canonical development email identity is ambiguous."
                )
            by_email[email] = person_id
    return by_name, by_email


def project_context_cases(
    p3_manifest: Mapping[str, Any], transition: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Project pre-gold suggestions through accepted canonical source affinities."""

    by_name, by_email = _person_indexes(transition)
    cases = []
    for result in p3_manifest.get("results") or []:
        join = result.get("join") if isinstance(result, Mapping) else None
        if not isinstance(join, Mapping):
            raise Plan0064DevelopmentReplayError("A development context join is invalid.")
        context_bundle = join.get("context_bundle")
        if not isinstance(context_bundle, Mapping):
            raise Plan0064DevelopmentReplayError(
                "A development context bundle is missing."
            )
        lineage_types = {
            str(item.get("evidence_id") or ""): str(item.get("source_type") or "")
            for item in context_bundle.get("lineage") or []
            if isinstance(item, Mapping) and item.get("evidence_id")
        }
        slots = []
        for outcome in join.get("review_outcomes") or []:
            if not isinstance(outcome, Mapping):
                raise Plan0064DevelopmentReplayError(
                    "A development review outcome is invalid."
                )
            mapped_people = set()
            for suggestion in outcome.get("suggestions") or []:
                if not isinstance(suggestion, Mapping):
                    continue
                email = _normalized(suggestion.get("email"))
                name = _normalized(suggestion.get("name"))
                person_id = by_email.get(email) if email else None
                person_id = person_id or by_name.get(name)
                if person_id:
                    mapped_people.add(person_id)
            evidence_ids = [
                str(value)
                for value in outcome.get("context_evidence_ids") or []
                if str(value)
            ]
            transcript_clues = sorted(
                value
                for value in evidence_ids
                if lineage_types.get(value) == "transcript_clue"
            )
            provenance = sorted(
                value
                for value in evidence_ids
                if lineage_types.get(value)
                and lineage_types.get(value) != "transcript_clue"
            )
            conflict = outcome.get("context_status") == "conflicting"
            candidate = (
                next(iter(mapped_people))
                if len(mapped_people) == 1
                and transcript_clues
                and provenance
                and not conflict
                else None
            )
            proposal = (
                {
                    "proposal_id": "development-context-"
                    + _hash(
                        {
                            "document_id": result["document_id"],
                            "speaker_ref": outcome["speaker_ref"],
                            "person_id": candidate,
                        }
                    )[:24],
                    "status": "candidate_match",
                    "prepared_person_id": candidate,
                    "transcript_clue_ids": transcript_clues,
                    "provenance_source_ids": provenance,
                    "factors": [],
                }
                if candidate
                else None
            )
            speaker_ref = f"{result['document_id']}::{outcome['speaker_ref']}"
            slots.append(
                {
                    "speaker_ref": speaker_ref,
                    "speaker_label": outcome["source_speaker_label"],
                    "disposition": "candidate" if candidate else "abstain",
                    "reason_code": (
                        "accepted_source_affinity_candidate"
                        if candidate
                        else "material_context_conflict"
                        if conflict
                        else "no_unique_accepted_source_affinity_candidate"
                    ),
                    "candidate_person_id": candidate,
                    "candidates": [proposal] if proposal else [],
                }
            )
        required_failures = [
            list(item)
            for item in context_bundle.get("source_failures") or []
            if isinstance(item, (list, tuple)) and len(item) == 3 and item[2] is True
        ]
        cases.append(
            {
                "document_id": result["document_id"],
                "speaker_slots": slots,
                "provider_failures": required_failures,
            }
        )
    return cases


def _score_development(
    *,
    p0_manifest: Mapping[str, Any],
    p4_manifest: Mapping[str, Any],
    p4_root: Path,
    thresholds: Mapping[str, float],
    adapters: Mapping[str, Any],
    score_fn: Callable[..., Mapping[str, Any]],
    decode_fn: Callable[[Path], array],
    profile_root: Path,
    reference_root: Path,
) -> list[dict[str, Any]]:
    clips = {
        str(item["slot_id"]): item for item in p4_manifest.get("audio_clips") or []
    }
    by_document: dict[str, list[dict[str, Any]]] = {}
    cards = p4_manifest["packet"]["cards"]
    for index, card in enumerate(cards, start=1):
        print(f"Scoring Plan 0064 development speaker {index}/{len(cards)}...", flush=True)
        slot_id = str(card["slot_id"])
        clip = clips[slot_id]
        clip_path = (
            p4_root.expanduser().absolute()
            / "preview"
            / str(clip["relative_path"])
        )
        scored = _score_slot(
            document_id=str(card["document_id"]),
            speaker=str(card["speaker_ref"]),
            probe=decode_fn(clip_path),
            profiles=p0_manifest["profile_inventory"]["active_profiles"],
            thresholds=thresholds,
            adapters=adapters,
            score_fn=score_fn,
            profile_root=profile_root.expanduser().absolute(),
            reference_root=reference_root.expanduser().absolute(),
        )
        by_document.setdefault(str(card["document_id"]), []).append(scored)
    return [
        {
            "document_id": document_id,
            "transcript_sha256": _hash(
                {"development_document_id": document_id, "kind": "reviewed-context"}
            ),
            "source_media_sha256": _hash(
                [clips[slot["speaker_ref"]]["sha256"] for slot in slots]
            ),
            "speaker_slots": slots,
        }
        for document_id, slots in by_document.items()
    ]


def build_development_gate(
    resolution: Sequence[Mapping[str, Any]], transition: Mapping[str, Any]
) -> dict[str, Any]:
    gold = {
        str(item["slot_id"]): str(item["person_id"])
        for item in transition.get("slot_bindings") or []
    }
    combined_correct = residual_correct = high_support_wrong = 0
    combined_candidate = residual_candidate = 0
    rows = []
    for recording in resolution:
        for slot in recording.get("speaker_slots") or []:
            slot_id = str(slot["speaker_ref"])
            target = gold.get(slot_id)
            condition_rows = []
            for condition in ("combined", "residual_policy"):
                view = slot[condition]
                proposed = (
                    str(view.get("candidate_person_id") or "")
                    if view.get("disposition") == "candidate"
                    else ""
                )
                correct = bool(proposed and target and proposed == target)
                wrong = bool(proposed and (not target or proposed != target))
                high_support = str(view.get("reason_code") or "") in HIGH_SUPPORT_REASONS
                residual_rule = (
                    condition == "residual_policy"
                    and view.get("reason_code")
                    == "two_known_plus_one_independently_supported_residual"
                )
                if condition == "combined":
                    combined_candidate += int(bool(proposed))
                    combined_correct += int(correct)
                elif residual_rule:
                    residual_candidate += int(bool(proposed))
                    residual_correct += int(correct)
                high_support_wrong += int(high_support and wrong)
                condition_rows.append(
                    {
                        "condition": condition,
                        "disposition": view.get("disposition"),
                        "reason_code": view.get("reason_code"),
                        "correct": correct,
                        "wrong": wrong,
                    }
                )
            rows.append(
                {
                    "speaker_ref": slot_id,
                    "gold_available": target is not None,
                    "conditions": condition_rows,
                }
            )
    core = {
        "schema_version": DEVELOPMENT_GATE_SCHEMA,
        "status": "reviewed_development_replay_complete",
        "source_corpus": "plan0063_reviewed_three_conversation",
        "source_disjoint_evaluation": False,
        "replay_exact": True,
        "recording_count": len(resolution),
        "speaker_slot_count": len(rows),
        "gold_named_slot_count": len(gold),
        "combined_candidate_count": combined_candidate,
        "combined_correct_count": combined_correct,
        "residual_candidate_count": residual_candidate,
        "residual_correct_count": residual_correct,
        "high_support_wrong_count": high_support_wrong,
        "quality_gate_passed": (
            high_support_wrong == 0
            and combined_correct >= 1
            and residual_correct >= 1
        ),
        "rows": rows,
        "action_counts": dict(ACTION_COUNTS),
    }
    return _content_addressed(core)


def execute_development_replay(
    p0_content_sha256: str,
    *,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    p3_root: Path = DEFAULT_PLAN0062_P3_ROOT,
    p4_root: Path = DEFAULT_PLAN0062_P4_ROOT,
    transition_path: Path = DEFAULT_PLAN0063_TRANSITION,
    threshold_application: Path = DEFAULT_THRESHOLD_APPLICATION,
    profile_root: Path = DEFAULT_PROFILE_ROOT,
    reference_root: Path = DEFAULT_REFERENCE_ROOT,
    score_fn: Callable[..., Mapping[str, Any]] = verification.score_profile,
    adapter_factory: Callable[[], Mapping[str, Any]] = verification.adapter_registry,
    decode_fn: Callable[[Path], array] = _decode,
) -> dict[str, Any]:
    preview = build_development_preview(
        p0_content_sha256,
        runtime_root=runtime_root,
        p3_root=p3_root,
        p4_root=p4_root,
        transition_path=transition_path,
        threshold_application=threshold_application,
    )
    run = runtime_root.expanduser().absolute() / (
        f"development-replay-{preview['content_sha256'][:24]}"
    )
    evidence_path, gate_path, receipt_path = (
        run / "private-evidence.json",
        run / "development-gate.json",
        run / "receipt.json",
    )
    if any(path.exists() for path in (evidence_path, gate_path, receipt_path)):
        return replay_development_replay(
            p0_content_sha256,
            runtime_root=runtime_root,
            p3_root=p3_root,
            p4_root=p4_root,
            transition_path=transition_path,
            threshold_application=threshold_application,
        )
    p0, p3, p4, transition = _private_sources(
        p0_content_sha256=p0_content_sha256,
        runtime_root=runtime_root,
        p3_root=p3_root,
        p4_root=p4_root,
        transition_path=transition_path,
    )
    adapters = {
        key: _CachingAdapter(value) for key, value in dict(adapter_factory()).items()
    }
    if set(adapters) != set(preview["candidate_ids"]):
        raise Plan0064DevelopmentReplayError(
            "The development adapter registry differs from P0."
        )
    thresholds = {
        item["candidate_id"]: float(item["threshold"])
        for item in preview["threshold_authority"]["units"]
    }
    acoustic = _score_development(
        p0_manifest=p0,
        p4_manifest=p4,
        p4_root=p4_root,
        thresholds=thresholds,
        adapters=adapters,
        score_fn=score_fn,
        decode_fn=decode_fn,
        profile_root=profile_root,
        reference_root=reference_root,
    )
    contexts = project_context_cases(p3, transition)
    if [item["document_id"] for item in acoustic] != [
        item["document_id"] for item in contexts
    ]:
        raise Plan0064DevelopmentReplayError(
            "Development acoustic and context recording order differs."
        )
    resolution = [
        resolve_conversation(acoustic_row, context_row)
        for acoustic_row, context_row in zip(acoustic, contexts, strict=True)
    ]
    gate = build_development_gate(resolution, transition)
    evidence = _content_addressed(
        {
            "schema_version": EVIDENCE_SCHEMA,
            "status": "complete_private_reviewed_development_replay",
            "preview_content_sha256": preview["content_sha256"],
            "recordings": resolution,
            "summary": {
                "recording_count": len(resolution),
                "speaker_slot_count": sum(
                    len(item["speaker_slots"]) for item in resolution
                ),
                "condition_disposition_counts": {
                    condition: dict(
                        sorted(
                            Counter(
                                slot[condition]["disposition"]
                                for item in resolution
                                for slot in item["speaker_slots"]
                            ).items()
                        )
                    )
                    for condition in (
                        "acoustic",
                        "context",
                        "combined",
                        "residual_policy",
                    )
                },
            },
            "contains_biometric_scores": True,
            "contains_raw_audio": False,
            "source_disjoint_evaluation": False,
            "action_counts": dict(ACTION_COUNTS),
        }
    )
    ensure_private_tree(run, run)
    write_immutable_private_json(evidence_path, evidence)
    write_immutable_private_json(gate_path, gate)
    receipt = _content_addressed(
        {
            "schema_version": RECEIPT_SCHEMA,
            "status": "development_replay_complete_zero_effect",
            "preview_content_sha256": preview["content_sha256"],
            "evidence_content_sha256": evidence["content_sha256"],
            "evidence_file_sha256": sha256_file(evidence_path),
            "development_gate_content_sha256": gate["content_sha256"],
            "development_gate_file_sha256": sha256_file(gate_path),
            "quality_gate_passed": gate["quality_gate_passed"],
            "summary": evidence["summary"],
            "action_counts": dict(ACTION_COUNTS),
        }
    )
    write_immutable_private_json(receipt_path, receipt)
    return {
        **receipt,
        "private_development_gate_path": str(gate_path),
        "private_receipt_path": str(receipt_path),
        "idempotent_replay": False,
    }


def replay_development_replay(
    p0_content_sha256: str,
    *,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    p3_root: Path = DEFAULT_PLAN0062_P3_ROOT,
    p4_root: Path = DEFAULT_PLAN0062_P4_ROOT,
    transition_path: Path = DEFAULT_PLAN0063_TRANSITION,
    threshold_application: Path = DEFAULT_THRESHOLD_APPLICATION,
) -> dict[str, Any]:
    preview = build_development_preview(
        p0_content_sha256,
        runtime_root=runtime_root,
        p3_root=p3_root,
        p4_root=p4_root,
        transition_path=transition_path,
        threshold_application=threshold_application,
    )
    run = runtime_root.expanduser().absolute() / (
        f"development-replay-{preview['content_sha256'][:24]}"
    )
    evidence_path, gate_path, receipt_path = (
        run / "private-evidence.json",
        run / "development-gate.json",
        run / "receipt.json",
    )
    for path in (evidence_path, gate_path, receipt_path):
        require_private_file(path, runtime_root.expanduser().absolute())
    evidence, gate, receipt = _read(evidence_path), _read(gate_path), _read(receipt_path)
    for value, label in (
        (evidence, "Development evidence"),
        (gate, "Development gate"),
        (receipt, "Development receipt"),
    ):
        _validate_hash(value, label=label)
    _p0, _p3, _p4, transition = _private_sources(
        p0_content_sha256=p0_content_sha256,
        runtime_root=runtime_root,
        p3_root=p3_root,
        p4_root=p4_root,
        transition_path=transition_path,
    )
    expected_gate = build_development_gate(evidence["recordings"], transition)
    if (
        evidence.get("preview_content_sha256") != preview["content_sha256"]
        or gate != expected_gate
        or receipt.get("preview_content_sha256") != preview["content_sha256"]
        or receipt.get("evidence_content_sha256") != evidence["content_sha256"]
        or receipt.get("evidence_file_sha256") != sha256_file(evidence_path)
        or receipt.get("development_gate_content_sha256") != gate["content_sha256"]
        or receipt.get("development_gate_file_sha256") != sha256_file(gate_path)
        or receipt.get("quality_gate_passed") != gate["quality_gate_passed"]
        or receipt.get("action_counts") != ACTION_COUNTS
    ):
        raise Plan0064DevelopmentReplayError(
            "The frozen Plan 0064 development replay drifted."
        )
    return {
        **receipt,
        "private_development_gate_path": str(gate_path),
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
        "preview": build_development_preview,
        "execute": execute_development_replay,
        "replay": replay_development_replay,
    }[args.action]
    result = action(args.p0_content_sha256, runtime_root=args.runtime_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
