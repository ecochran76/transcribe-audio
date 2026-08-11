#!/usr/bin/env python3
"""Freeze Plan 0065 D0 development and exposure authority without effects."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.request import urlopen

from acoustic_audio_derivatives import (
    canonical_artifact_hash,
    ensure_private_tree,
    read_private_object,
    require_private_file,
    sha256_file,
    write_immutable_private_json,
)
import speaker_identity_plan0064_p0 as p0
import speaker_identity_plan0064_p4_measurement as p4m


DEFAULT_RUNTIME_ROOT = Path("~/.local/state/transcribe-audio/plan-0065")
DEFAULT_PLAN0064_ROOT = p0.DEFAULT_RUNTIME_ROOT
DEFAULT_API_BASE_URL = "http://127.0.0.1:18876"

MANIFEST_SCHEMA = "transcribe-audio.plan0065-d0-diagnostic-manifest.v1"
RECEIPT_SCHEMA = "transcribe-audio.plan0065-d0-diagnostic-receipt.v1"
EXPOSURE_SCHEMA = "transcribe-audio.plan0065-development-exposure-set.v1"
INVENTORY_SCHEMA = "transcribe-audio.plan0065-current-profile-inventory.v1"
PROVIDER_SCHEMA = "transcribe-audio.plan0065-provider-readiness.v1"

P0_CONTENT_SHA256 = "f24722166f5f147ee6b26b13bba87d1f12ab60530c3ca0add3d8687046c5675a"
P1_PREVIEW_SHA256 = "8dc9ef4e6cde703869480409b9f7f0f3bda1489636671dfb27555c04369006e2"
P1_CONTENT_SHA256 = "b6a87465ddcdef0a781554c56cf1fe8bdad6b86c8ac7b5ae2905300db320bbb4"
P1_RECEIPT_SHA256 = "d0e2441adbbaadf22fd401946a36a858317d3f0f36a26a72607e7f2973407a30"
P2_PREVIEW_SHA256 = "d6014903bf89a4398d3fd392b9feae65d9105c093f21264d954a2649c5253a23"
P2_RECEIPT_SHA256 = "50a7f4fd15b8c65c1faf4628309e72796661ac7760651eb7c9666d9117d9bd6b"
P3_PREVIEW_SHA256 = "2ec73512fc8122efd79201471473b9ac6f5e7f1197f4a5a9c644eebe1537a55b"
P3_CONTENT_SHA256 = "2f55e7adb9a48e44073e402bd3bc802ddc10c518cdb3d158d00f5a5058492dcb"
P3_RECEIPT_SHA256 = "b630d12d6ce21804d8cd0ad4e24ff6f22730ad365c0ea271f9e2db6d661d115e"
REVIEW_PREVIEW_SHA256 = "87a58a64d82270cb4585402ee9de2e97bc380cda3b75a5b5b2d70c6a4d54df46"
REVIEW_AUTHORITY_SHA256 = "e2df49c9fb081ea50d17d77a09b8c26a577b0e6f3cb3b64d8acb580e7b8a0daf"
REVIEW_RECEIPT_SHA256 = "22bbfd4eb3559801af97dab4f94dbb1d79c559820e1297a74299e77a79612680"
AUTHORITY_BRIDGE_SHA256 = "031ce0f0f2864e3c34a1ff081644629557395f4a948f06b0c2d9c2f3179ea67d"
GOLD_SHA256 = "1645c31a647a5632a0929870d6c442d8cca184cb64bf21c47ea40bca44da6368"
MEASUREMENT_SHA256 = "baa26f05bee01165ddf9f5dd77de39b47cc1da9be71fbda5568a73673f8c09c7"
TERMINAL_SHA256 = "f178f4187d0e8c877362310563738144854508fb4acba8b3ea227b79e829d5b6"
DEVELOPMENT_GATE_SHA256 = "cb942cbd9efea0bdfc64a633a8e8aa179149d6ed67beb1dbd780fb01e132b0c1"

ACTION_COUNTS = {
    "speaker_assignments": 0,
    "new_enrollments": 0,
    "profile_mutations": 0,
    "reference_mutations": 0,
    "threshold_default_changes": 0,
    "knowledge_writes": 0,
    "graphiti_writes": 0,
    "provider_writes": 0,
    "external_writes": 0,
    "historical_reprocessing": 0,
    "model_turns": 0,
}


class Plan0065D0Error(ValueError):
    """Raised when Plan 0065 D0 authority is stale, incomplete, or unsafe."""


def _hash(value: Any) -> str:
    return canonical_artifact_hash(value)


def _content_addressed(value: Mapping[str, Any]) -> dict[str, Any]:
    core = dict(value)
    core.pop("content_sha256", None)
    return {**core, "content_sha256": _hash(core)}


def _validate_content_addressed(value: Mapping[str, Any], *, label: str) -> None:
    core = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != _hash(core):
        raise Plan0065D0Error(f"{label} content hash drifted.")


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode:
        raise Plan0065D0Error(result.stderr.strip() or "Git authority read failed.")
    return result.stdout.strip()


def repository_authority() -> dict[str, Any]:
    """Bind D0 to the latest committed module while allowing later commits."""

    module = Path(__file__).resolve()
    root = Path(_git("rev-parse", "--show-toplevel")).resolve()
    try:
        relative = module.relative_to(root).as_posix()
    except ValueError as exc:
        raise Plan0065D0Error("The D0 module is outside the repository.") from exc
    status = _git("status", "--porcelain=v1")
    upstream = _git("rev-parse", "@{upstream}")
    ahead = int(_git("rev-list", "--count", f"{upstream}..HEAD"))
    behind = int(_git("rev-list", "--count", f"HEAD..{upstream}"))
    module_commit = _git("log", "-1", "--format=%H", "--", relative)
    if not module_commit:
        raise Plan0065D0Error("The D0 module has no committed authority.")
    committed = subprocess.run(
        ["git", "show", f"{module_commit}:{relative}"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if committed.returncode:
        raise Plan0065D0Error("The committed D0 module cannot be read.")
    module_sha256 = hashlib.sha256(module.read_bytes()).hexdigest()
    committed_sha256 = hashlib.sha256(committed.stdout).hexdigest()
    authority = {
        "module_name": relative,
        "module_commit": module_commit,
        "module_sha256": module_sha256,
        "module_blob_matches": module_sha256 == committed_sha256,
        "clean": not status,
        "upstream_ahead": ahead,
        "upstream_behind": behind,
    }
    if (
        authority["module_blob_matches"] is not True
        or authority["clean"] is not True
        or ahead
        or behind
    ):
        raise Plan0065D0Error("D0 repository authority is not clean and upstream-even.")
    return authority


def _read_bound(path: Path, root: Path, *, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    require_private_file(path, root)
    value = read_private_object(path)
    if not isinstance(value, dict):
        raise Plan0065D0Error(f"{label} is not an object.")
    if "content_sha256" in value:
        _validate_content_addressed(value, label=label)
    return value, {
        "label": label,
        "path": str(path),
        "file_sha256": sha256_file(path),
        "content_sha256": value.get("content_sha256"),
        "schema_version": value.get("schema_version"),
    }


def _expected(value: Mapping[str, Any], expected: str, *, label: str) -> None:
    if value.get("content_sha256") != expected:
        raise Plan0065D0Error(f"{label} is not the frozen Plan 0064 authority.")


def _plan0064_paths(root: Path) -> dict[str, Path]:
    base = root.expanduser().absolute()
    return {
        "root": base,
        "p0": base / f"p0-{P0_CONTENT_SHA256[:24]}",
        "p1": base / f"p1-{P1_PREVIEW_SHA256[:24]}",
        "p2": base / f"p2-{P2_PREVIEW_SHA256[:24]}",
        "p3": base / f"p3-{P3_PREVIEW_SHA256[:24]}",
        "review": base / f"p4-review-{REVIEW_PREVIEW_SHA256[:24]}",
        "submission": base / "p4-submission-6df988b11c152b78f9da59ab",
        "measurement": base / f"p4-measurement-{GOLD_SHA256[:24]}",
        "development": base / "development-replay-a2a3e65cd1a35531348c1795",
    }


def build_exposure_set(
    *,
    p0_manifest: Mapping[str, Any],
    p1_evidence: Mapping[str, Any],
    p2_cases: Sequence[Mapping[str, Any]],
    p3_resolution: Mapping[str, Any],
    review_authority: Mapping[str, Any],
    gold: Mapping[str, Any],
    measurement: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the permanent development exclusion without copying identities."""

    selected = [
        item
        for item in p0_manifest.get("evaluation_cohort", {}).get("considered", [])
        if item.get("disposition") == "selected_evaluation_candidate"
    ]
    p1_slots = [
        slot
        for recording in p1_evidence.get("recordings") or []
        for slot in recording.get("speaker_slots") or []
    ]
    p2_by_document = {
        str(case.get("document_id") or ""): case
        for case in p2_cases
    }
    p1_document_ids = [
        str(recording.get("document_id") or "")
        for recording in p1_evidence.get("recordings") or []
    ]
    if (
        len(p2_by_document) != len(p2_cases)
        or set(p2_by_document) != set(p1_document_ids)
    ):
        raise Plan0065D0Error("Plan 0064 P2 recording set does not align.")
    p2_slots = [
        slot
        for document_id in p1_document_ids
        for slot in p2_by_document[document_id].get("speaker_slots") or []
    ]
    p3_slots = [
        slot
        for recording in p3_resolution.get("recordings") or []
        for slot in recording.get("speaker_slots") or []
    ]
    cases = list(review_authority.get("cases") or [])
    decisions = list(gold.get("decisions") or [])
    rows = list(measurement.get("rows") or [])
    expected_refs = [str(item.get("speaker_ref") or "") for item in p1_slots]
    if (
        len(selected) != 12
        or len(expected_refs) != 39
        or expected_refs != [str(item.get("speaker_ref") or "") for item in p2_slots]
        or expected_refs != [str(item.get("speaker_ref") or "") for item in p3_slots]
        or expected_refs != [str(item.get("speaker_ref") or "") for item in cases]
        or expected_refs != [str(item.get("speaker_ref") or "") for item in decisions]
        or expected_refs != [str(item.get("speaker_ref") or "") for item in rows]
    ):
        raise Plan0065D0Error("Plan 0064 exposure denominators do not align.")

    reference_inventory = p0_manifest.get("reference_inventory") or {}
    prior_exposure = p0_manifest.get("prior_identity_exposure") or {}
    development_sources = list(reference_inventory.get("development_sources") or [])
    development_recording_hashes = sorted(
        set(reference_inventory.get("development_recording_hashes") or [])
        | {str(item["source_media_sha256"]) for item in selected}
    )
    document_ids = sorted(
        set(prior_exposure.get("document_ids") or [])
        | {str(item["document_id"]) for item in selected}
    )
    full_recordings = [
        {
            "document_id": str(item["document_id"]),
            "source_media_sha256": str(item["source_media_sha256"]),
            "transcript_sha256": str(item["artifact_sha256"]),
            "speaker_labels": list(item["speaker_labels"]),
            "window_policy": "full_recording",
        }
        for item in selected
    ]
    source_windows = [
        {
            "source_sha256": str(item["source_sha256"]),
            "start_seconds": float(item["start_seconds"]),
            "end_seconds": float(item["end_seconds"]),
        }
        for item in development_sources
    ]
    clip_rows = [
        {
            "speaker_ref": str(item["speaker_ref"]),
            "clip_sha256": str(item["clip_sha256"]),
            "recording_filename": str(item["recording_filename"]),
        }
        for item in cases
    ]
    probe_hashes = sorted(str(item.get("probe_sha256") or "") for item in p1_slots)
    decision_rows = [
        {
            "speaker_ref": str(item["speaker_ref"]),
            "decision": str(item["decision"]),
            "person_id": item.get("person_id"),
            "note": str(item.get("note") or ""),
        }
        for item in decisions
    ]
    return _content_addressed(
        {
            "schema_version": EXPOSURE_SCHEMA,
            "status": "plan0064_and_prior_sources_permanently_development_only",
            "recording_hashes": development_recording_hashes,
            "recording_hash_set_sha256": _hash(development_recording_hashes),
            "document_ids": document_ids,
            "document_id_set_sha256": _hash(document_ids),
            "full_recordings": full_recordings,
            "full_recording_set_sha256": _hash(full_recordings),
            "source_windows": source_windows,
            "source_window_set_sha256": _hash(source_windows),
            "review_clips": clip_rows,
            "review_clip_set_sha256": _hash(clip_rows),
            "probe_hashes": probe_hashes,
            "probe_hash_set_sha256": _hash(probe_hashes),
            "decision_rows": decision_rows,
            "decision_row_set_sha256": _hash(decision_rows),
            "action_counts": dict(ACTION_COUNTS),
        }
    )


def current_profile_inventory() -> dict[str, Any]:
    """Read the current governed reference/profile/person-binding matrix."""

    references = p0._reference_inventory(p0.DEFAULT_REFERENCE_ROOT)
    profiles = p0._profile_inventory(p0.DEFAULT_PROFILE_ROOT, references)
    bindings = p0._canonical_bindings(p0.DEFAULT_TRANSCRIPT_ROOT, profiles)
    binding_by_subject = {
        str(item["acoustic_subject_id"]): item
        for item in bindings["subject_bindings"]
    }
    active_profiles = []
    for profile in profiles["active_profiles"]:
        binding = binding_by_subject[str(profile["person_ref_id"])]
        active_profiles.append(
            {
                **profile,
                "canonical_person_id": binding["person_id"],
                "binding_status": binding["binding_status"],
                "identity_candidate_eligible": binding[
                    "identity_candidate_eligible"
                ],
            }
        )
    profiles = {**profiles, "active_profiles": active_profiles}
    profiles["identity_ready_profile_count"] = sum(
        bool(item["identity_candidate_eligible"]) for item in active_profiles
    )
    profiles["unbound_active_profile_count"] = sum(
        not item["identity_candidate_eligible"] for item in active_profiles
    )
    profiles["active_profile_set_sha256"] = _hash(active_profiles)
    return _content_addressed(
        {
            "schema_version": INVENTORY_SCHEMA,
            "status": "current_governed_profile_inventory_read_only",
            "reference_inventory": references,
            "profile_inventory": profiles,
            "canonical_bindings": bindings,
            "action_counts": dict(ACTION_COUNTS),
        }
    )


def _get_json(url: str) -> dict[str, Any]:
    with urlopen(url, timeout=10) as response:
        value = json.loads(response.read())
    if not isinstance(value, dict):
        raise Plan0065D0Error("The local transcript API returned a non-object.")
    return value


def provider_readiness(*, base_url: str = DEFAULT_API_BASE_URL) -> dict[str, Any]:
    """Read sanitized speaker-route readiness without sending a model turn."""

    provider_payload = _get_json(f"{base_url.rstrip('/')}/api/intelligence/providers")
    config_payload = _get_json(f"{base_url.rstrip('/')}/api/intelligence/config")
    providers = {
        str(item.get("id") or item.get("provider_id") or item.get("name") or ""): item
        for item in provider_payload.get("providers") or []
        if isinstance(item, Mapping)
    }
    task_config = config_payload.get("tasks") or {}
    if isinstance(task_config, Mapping):
        configured = task_config.get("speaker_disambiguation")
        tasks = [configured] if isinstance(configured, Mapping) else []
    else:
        tasks = [
            item
            for item in task_config
            if isinstance(item, Mapping)
            and str(item.get("task") or item.get("task_id") or item.get("id") or "")
            == "speaker_disambiguation"
        ]
    if len(tasks) != 1:
        raise Plan0065D0Error("The speaker_disambiguation task route is ambiguous.")
    task = tasks[0]
    provider_id = str(task.get("provider") or "")
    selected = providers.get(provider_id)
    if not provider_id or not isinstance(selected, Mapping):
        raise Plan0065D0Error("The selected speaker provider is not registered.")
    fallback_ids = list(task.get("fallbacks") or [])
    if task.get("fallback_provider") and not fallback_ids:
        fallback_ids = [task["fallback_provider"]]
    if len(fallback_ids) > 1:
        raise Plan0065D0Error("D0 permits at most one contextual fallback route.")
    fallback_id = str(fallback_ids[0]) if fallback_ids else None
    fallback = providers.get(fallback_id) if fallback_id else None
    sanitized_provider = {
        "provider_id": provider_id,
        "status": str(selected.get("status") or "unknown"),
        "ready": selected.get("ready") is True,
        "capabilities": selected.get("capabilities"),
    }
    sanitized_fallback = (
        {
            "provider_id": fallback_id,
            "status": str((fallback or {}).get("status") or "unknown"),
            "ready": (fallback or {}).get("ready") is True,
            "capabilities": (fallback or {}).get("capabilities"),
        }
        if fallback_id
        else None
    )
    return _content_addressed(
        {
            "schema_version": PROVIDER_SCHEMA,
            "status": (
                "ready_for_bounded_context_execution"
                if sanitized_provider["ready"]
                else "provider_not_ready"
            ),
            "task": "speaker_disambiguation",
            "model": str(task.get("model") or ""),
            "primary": sanitized_provider,
            "fallback": sanitized_fallback,
            "max_requests_per_case": {
                "primary": 1,
                "fallback": 1 if fallback_id else 0,
            },
            "did_start_session": False,
            "did_send_model_turn": False,
            "action_counts": dict(ACTION_COUNTS),
        }
    )


def collect_plan0064_authority(
    *, plan0064_root: Path = DEFAULT_PLAN0064_ROOT
) -> dict[str, Any]:
    """Validate and bind the complete closed Plan 0064 development lineage."""

    paths = _plan0064_paths(plan0064_root)
    root = paths["root"]
    bindings: list[dict[str, Any]] = []

    def read(path: Path, label: str) -> dict[str, Any]:
        value, binding = _read_bound(path, root, label=label)
        bindings.append(binding)
        return value

    p0_manifest = read(paths["p0"] / "private-manifest.json", "Plan 0064 P0 manifest")
    p0_receipt = read(paths["p0"] / "receipt.json", "Plan 0064 P0 receipt")
    if p0._validate_manifest(p0_manifest) != P0_CONTENT_SHA256:
        raise Plan0065D0Error("Plan 0064 P0 manifest authority drifted.")
    if p0_receipt.get("manifest_file_sha256") != sha256_file(
        paths["p0"] / "private-manifest.json"
    ):
        raise Plan0065D0Error("Plan 0064 P0 receipt lost its file binding.")

    p1_evidence = read(
        paths["p1"] / "private-acoustic-evidence.json", "Plan 0064 P1 evidence"
    )
    p1_receipt = read(paths["p1"] / "receipt.json", "Plan 0064 P1 receipt")
    _expected(p1_evidence, P1_CONTENT_SHA256, label="Plan 0064 P1 evidence")
    _expected(p1_receipt, P1_RECEIPT_SHA256, label="Plan 0064 P1 receipt")

    p2_cases = [
        read(path, f"Plan 0064 P2 case {path.stem}")
        for path in sorted((paths["p2"] / "cases").glob("*.json"))
    ]
    p2_receipt = read(paths["p2"] / "receipt.json", "Plan 0064 P2 receipt")
    _expected(p2_receipt, P2_RECEIPT_SHA256, label="Plan 0064 P2 receipt")
    if sorted(item["content_sha256"] for item in p2_cases) != sorted(
        p2_receipt.get("case_content_sha256s") or []
    ):
        raise Plan0065D0Error("Plan 0064 P2 case set drifted.")

    p3_resolution = read(
        paths["p3"] / "private-resolution.json", "Plan 0064 P3 resolution"
    )
    p3_receipt = read(paths["p3"] / "receipt.json", "Plan 0064 P3 receipt")
    _expected(p3_resolution, P3_CONTENT_SHA256, label="Plan 0064 P3 resolution")
    _expected(p3_receipt, P3_RECEIPT_SHA256, label="Plan 0064 P3 receipt")

    review_authority = read(
        paths["review"] / "review-authority.json", "Plan 0064 P4 review authority"
    )
    review_receipt = read(paths["review"] / "receipt.json", "Plan 0064 P4 review receipt")
    _expected(review_authority, REVIEW_AUTHORITY_SHA256, label="Plan 0064 P4 review authority")
    _expected(review_receipt, REVIEW_RECEIPT_SHA256, label="Plan 0064 P4 review receipt")

    bridge = read(paths["submission"] / "authority-bridge.json", "Plan 0064 authority bridge")
    read(paths["submission"] / "submitted-decisions.json", "Plan 0064 source decisions")
    read(paths["submission"] / "rebound-decisions.json", "Plan 0064 rebound decisions")
    _expected(bridge, AUTHORITY_BRIDGE_SHA256, label="Plan 0064 authority bridge")

    gold = read(paths["measurement"] / "human-gold.json", "Plan 0064 human gold")
    measurement = read(paths["measurement"] / "measurement.json", "Plan 0064 measurement")
    terminal = read(paths["measurement"] / "terminal.json", "Plan 0064 terminal")
    development_gate = read(
        paths["development"] / "development-gate.json",
        "Plan 0064 development gate",
    )
    _expected(gold, GOLD_SHA256, label="Plan 0064 human gold")
    _expected(measurement, MEASUREMENT_SHA256, label="Plan 0064 measurement")
    _expected(terminal, TERMINAL_SHA256, label="Plan 0064 terminal")
    _expected(development_gate, DEVELOPMENT_GATE_SHA256, label="Plan 0064 development gate")
    replay = p4m.replay_human_gold_and_measurement(
        gold_content_sha256=GOLD_SHA256,
        p0_content_sha256=P0_CONTENT_SHA256,
        runtime_root=root,
        development_gate=development_gate,
    )
    if (
        replay.get("content_sha256") != TERMINAL_SHA256
        or replay.get("terminal_decision") != "withhold_p5"
        or replay.get("idempotent_replay") is not True
        or any((replay.get("action_counts") or {}).values())
    ):
        raise Plan0065D0Error("Plan 0064 terminal did not replay exactly.")

    exposure = build_exposure_set(
        p0_manifest=p0_manifest,
        p1_evidence=p1_evidence,
        p2_cases=p2_cases,
        p3_resolution=p3_resolution,
        review_authority=review_authority,
        gold=gold,
        measurement=measurement,
    )
    case_status_counts = dict(sorted(Counter(item["status"] for item in p2_cases).items()))
    failure_stage_counts = dict(
        sorted(Counter(str(item.get("failure_stage") or "none") for item in p2_cases).items())
    )
    return _content_addressed(
        {
            "schema_version": "transcribe-audio.plan0065-plan0064-authority.v1",
            "status": "closed_plan0064_replayed_as_development_only",
            "lineage": {
                "p0_content_sha256": P0_CONTENT_SHA256,
                "p1_content_sha256": P1_CONTENT_SHA256,
                "p2_receipt_content_sha256": P2_RECEIPT_SHA256,
                "p3_content_sha256": P3_CONTENT_SHA256,
                "review_authority_content_sha256": REVIEW_AUTHORITY_SHA256,
                "authority_bridge_content_sha256": AUTHORITY_BRIDGE_SHA256,
                "human_gold_content_sha256": GOLD_SHA256,
                "measurement_content_sha256": MEASUREMENT_SHA256,
                "development_gate_content_sha256": DEVELOPMENT_GATE_SHA256,
                "terminal_content_sha256": TERMINAL_SHA256,
                "terminal_decision": "withhold_p5",
            },
            "artifact_bindings": bindings,
            "artifact_binding_set_sha256": _hash(bindings),
            "exposure_set": exposure,
            "p2_case_status_counts": case_status_counts,
            "p2_failure_stage_counts": failure_stage_counts,
            "action_counts": dict(ACTION_COUNTS),
        }
    )


def build_d0_manifest(
    *,
    plan0064_authority: Mapping[str, Any],
    inventory: Mapping[str, Any],
    provider: Mapping[str, Any],
    repository: Mapping[str, Any],
) -> dict[str, Any]:
    for value, label in (
        (plan0064_authority, "Plan 0064 development authority"),
        (inventory, "Current profile inventory"),
        (provider, "Provider readiness"),
    ):
        _validate_content_addressed(value, label=label)
        if any((value.get("action_counts") or {}).values()):
            raise Plan0065D0Error(f"{label} carries an effect.")
    if (
        plan0064_authority.get("lineage", {}).get("terminal_content_sha256")
        != TERMINAL_SHA256
        or plan0064_authority.get("lineage", {}).get("terminal_decision")
        != "withhold_p5"
        or provider.get("did_start_session") is not False
        or provider.get("did_send_model_turn") is not False
        or repository.get("clean") is not True
        or repository.get("module_blob_matches") is not True
        or int(repository.get("upstream_ahead") or 0)
        or int(repository.get("upstream_behind") or 0)
    ):
        raise Plan0065D0Error("D0 activation authority is incomplete or unsafe.")
    return _content_addressed(
        {
            "schema_version": MANIFEST_SCHEMA,
            "status": "d0_development_exposure_authority_frozen",
            "repository_authority": dict(repository),
            "plan0064_authority": dict(plan0064_authority),
            "current_profile_inventory": dict(inventory),
            "provider_readiness": dict(provider),
            "ready_packets": ["d1_acoustic_safety", "d2_contextual_evidence"],
            "blocked_packets": ["d3_joined_residual", "e0_fresh_authority"],
            "human_gate": "complete_literal_gold_required_for_any_new_review",
            "action_counts": dict(ACTION_COUNTS),
        }
    )


def _paths(runtime_root: Path, content_sha256: str) -> dict[str, Path]:
    root = runtime_root.expanduser().absolute()
    run = root / f"d0-{content_sha256[:24]}"
    return {
        "root": root,
        "run": run,
        "manifest": run / "private-manifest.json",
        "receipt": run / "receipt.json",
    }


def preview_d0(
    *,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    plan0064_root: Path = DEFAULT_PLAN0064_ROOT,
    api_base_url: str = DEFAULT_API_BASE_URL,
) -> dict[str, Any]:
    del runtime_root
    return build_d0_manifest(
        plan0064_authority=collect_plan0064_authority(
            plan0064_root=plan0064_root
        ),
        inventory=current_profile_inventory(),
        provider=provider_readiness(base_url=api_base_url),
        repository=repository_authority(),
    )


def freeze_d0(
    *,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    plan0064_root: Path = DEFAULT_PLAN0064_ROOT,
    api_base_url: str = DEFAULT_API_BASE_URL,
) -> dict[str, Any]:
    manifest = preview_d0(
        runtime_root=runtime_root,
        plan0064_root=plan0064_root,
        api_base_url=api_base_url,
    )
    paths = _paths(runtime_root, manifest["content_sha256"])
    if paths["receipt"].exists():
        return replay_d0(
            manifest_content_sha256=manifest["content_sha256"],
            runtime_root=runtime_root,
            plan0064_root=plan0064_root,
            api_base_url=api_base_url,
        )
    if paths["run"].exists():
        raise Plan0065D0Error("A partial D0 runtime directory exists.")
    ensure_private_tree(paths["root"], paths["run"])
    write_immutable_private_json(paths["manifest"], manifest)
    exposure = manifest["plan0064_authority"]["exposure_set"]
    inventory = manifest["current_profile_inventory"]
    provider = manifest["provider_readiness"]
    receipt = _content_addressed(
        {
            "schema_version": RECEIPT_SCHEMA,
            "status": "d0_frozen_zero_effect",
            "manifest_content_sha256": manifest["content_sha256"],
            "manifest_file_sha256": sha256_file(paths["manifest"]),
            "recording_hash_count": len(exposure["recording_hashes"]),
            "recording_hash_set_sha256": exposure["recording_hash_set_sha256"],
            "source_window_count": len(exposure["source_windows"]),
            "source_window_set_sha256": exposure["source_window_set_sha256"],
            "review_clip_count": len(exposure["review_clips"]),
            "decision_count": len(exposure["decision_rows"]),
            "active_reference_count": len(
                inventory["reference_inventory"]["active_references"]
            ),
            "active_profile_count": len(
                inventory["profile_inventory"]["active_profiles"]
            ),
            "identity_ready_profile_count": inventory["profile_inventory"][
                "identity_ready_profile_count"
            ],
            "profile_inventory_content_sha256": inventory["content_sha256"],
            "provider_readiness_content_sha256": provider["content_sha256"],
            "provider_status": provider["status"],
            "ready_packets": list(manifest["ready_packets"]),
            "action_counts": dict(ACTION_COUNTS),
        }
    )
    write_immutable_private_json(paths["receipt"], receipt)
    return {
        **receipt,
        "private_manifest_path": str(paths["manifest"]),
        "private_receipt_path": str(paths["receipt"]),
        "idempotent_replay": False,
    }


def replay_d0(
    *,
    manifest_content_sha256: str,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    plan0064_root: Path = DEFAULT_PLAN0064_ROOT,
    api_base_url: str = DEFAULT_API_BASE_URL,
) -> dict[str, Any]:
    paths = _paths(runtime_root, manifest_content_sha256)
    require_private_file(paths["manifest"], paths["root"])
    require_private_file(paths["receipt"], paths["root"])
    manifest = read_private_object(paths["manifest"])
    receipt = read_private_object(paths["receipt"])
    expected = preview_d0(
        runtime_root=runtime_root,
        plan0064_root=plan0064_root,
        api_base_url=api_base_url,
    )
    _validate_content_addressed(manifest, label="D0 manifest")
    _validate_content_addressed(receipt, label="D0 receipt")
    if (
        manifest != expected
        or manifest.get("content_sha256") != manifest_content_sha256
        or receipt.get("schema_version") != RECEIPT_SCHEMA
        or receipt.get("manifest_content_sha256") != manifest_content_sha256
        or receipt.get("manifest_file_sha256") != sha256_file(paths["manifest"])
        or any((receipt.get("action_counts") or {}).values())
    ):
        raise Plan0065D0Error("The frozen D0 authority drifted.")
    return {
        **receipt,
        "private_manifest_path": str(paths["manifest"]),
        "private_receipt_path": str(paths["receipt"]),
        "idempotent_replay": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preview", "freeze", "replay"))
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--plan0064-root", type=Path, default=DEFAULT_PLAN0064_ROOT)
    parser.add_argument("--api-base-url", default=DEFAULT_API_BASE_URL)
    parser.add_argument("--manifest-content-sha256")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.mode == "preview":
        result = preview_d0(
            runtime_root=args.runtime_root,
            plan0064_root=args.plan0064_root,
            api_base_url=args.api_base_url,
        )
    elif args.mode == "freeze":
        result = freeze_d0(
            runtime_root=args.runtime_root,
            plan0064_root=args.plan0064_root,
            api_base_url=args.api_base_url,
        )
    else:
        if not args.manifest_content_sha256:
            raise SystemExit("replay requires --manifest-content-sha256")
        result = replay_d0(
            manifest_content_sha256=args.manifest_content_sha256,
            runtime_root=args.runtime_root,
            plan0064_root=args.plan0064_root,
            api_base_url=args.api_base_url,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
