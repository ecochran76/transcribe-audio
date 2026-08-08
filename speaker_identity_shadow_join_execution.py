from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import acoustic_plan0057
import provenance_config
from acoustic_audio_derivatives import (
    canonical_artifact_hash,
    ensure_private_tree,
    read_private_object,
    require_private_file,
    sha256_file,
    write_immutable_private_json,
)
from speaker_identity_evidence_execution import (
    _acoustic_review_for_case,
    _contact_rows,
    _context_for_case,
    _model_versions,
    _source_case,
    _stable_id,
    adapt_acoustic_review,
    normalize_explicit_provider_scopes,
)
from speaker_identity_orchestration import (
    IdentityOrchestrationError,
    _fail,
    _quick_check,
    _sqlite_backup,
    negative_action_vector,
    replay_activation,
    replay_shadow_store,
    validate_bundle_bindings,
)


ACTIVATION_VERSION = "transcribe-audio.plan0060-activation.v1"
P2A_MANIFEST_VERSION = "transcribe-audio.plan0060-p2a-acoustic-manifest.v1"
P2A_RECEIPT_VERSION = "transcribe-audio.plan0060-p2a-acoustic-receipt.v1"
P2B_MANIFEST_VERSION = "transcribe-audio.plan0060-p2b-context-manifest.v1"
P2B_RECEIPT_VERSION = "transcribe-audio.plan0060-p2b-context-receipt.v1"
EXPECTED_RECORDINGS = 3
EXPECTED_SPEAKERS = 10


def _lane_paths(runtime_root: Path, activation_sha256: str, lane: str) -> dict[str, Path]:
    if lane not in {"p2a-acoustic", "p2b-context"}:
        _fail("invalid_plan0060_lane", "Plan 0060 lane is not recognized.")
    root = runtime_root.expanduser().absolute()
    run = root / f"{lane}-{activation_sha256[:24]}"
    return {
        "root": root,
        "run": run,
        "sources": run / "sources",
        "context_root": run / "context-shadow",
        "context_database": run / "context-shadow" / "transcripts.sqlite3",
        "manifest": run / "private-manifest.json",
        "receipt": run / "receipt.json",
    }


def _activation_paths(runtime_root: Path, activation_sha256: str) -> tuple[Path, Path]:
    run = runtime_root.expanduser().absolute() / f"activation-{activation_sha256[:24]}"
    return run / "private-manifest.json", run / "receipt.json"


def replay_plan0060_activation(
    *, runtime_root: Path, activation_sha256: str
) -> dict[str, Any]:
    root = runtime_root.expanduser().absolute()
    manifest_path, receipt_path = _activation_paths(root, activation_sha256)
    require_private_file(manifest_path, root)
    require_private_file(receipt_path, root)
    manifest = read_private_object(manifest_path)
    receipt = read_private_object(receipt_path)
    if (
        manifest.get("schema_version") != ACTIVATION_VERSION
        or manifest.get("status") != "activated_pre_implementation"
        or canonical_artifact_hash(manifest) != activation_sha256
        or receipt.get("content_sha256") != activation_sha256
        or receipt.get("manifest_sha256") != sha256_file(manifest_path)
        or int(receipt.get("recording_count") or 0) != EXPECTED_RECORDINGS
        or int(receipt.get("speaker_ref_count") or 0) != EXPECTED_SPEAKERS
        or receipt.get("inherited_replay_verified") is not True
        or receipt.get("negative_actions_preserved") is not True
        or any((manifest.get("negative_actions") or {}).values())
    ):
        _fail("plan0060_activation_invalid", "Plan 0060 activation binding is invalid.")
    return {
        "manifest": manifest,
        "receipt": receipt,
        "manifest_path": str(manifest_path),
        "receipt_path": str(receipt_path),
        "idempotent_replay": True,
    }


def _shared_inputs(
    *,
    runtime_root: Path,
    activation_sha256: str,
    plan0059_runtime_root: Path,
    plan0059_activation_sha256: str,
    plan0057_authority_manifest: Path,
) -> dict[str, Any]:
    plan0060 = replay_plan0060_activation(
        runtime_root=runtime_root,
        activation_sha256=activation_sha256,
    )
    activation = replay_activation(
        plan0059_activation_sha256,
        runtime_root=plan0059_runtime_root,
    )
    p1 = replay_shadow_store(
        activation_content_sha256=plan0059_activation_sha256,
        runtime_root=plan0059_runtime_root,
    )
    inherited = plan0060["manifest"].get("inherited") or {}
    if (
        inherited.get("plan0059_activation_content_sha256")
        != plan0059_activation_sha256
        or inherited.get("plan0059_p1_content_sha256") != p1.get("content_sha256")
        or activation.get("recording_count") != EXPECTED_RECORDINGS
        or activation.get("speaker_ref_count") != EXPECTED_SPEAKERS
    ):
        _fail("plan0059_inheritance_drift", "Inherited Plan 0059 evidence drifted.")

    p1_manifest_path = Path(str(p1["manifest_path"]))
    require_private_file(p1_manifest_path, plan0059_runtime_root)
    p1_manifest = read_private_object(p1_manifest_path)
    active_shadow_root = Path(str(p1["active_shadow_root"]))
    active_database = active_shadow_root / "transcripts.sqlite3"
    if _quick_check(active_database) != "ok":
        _fail("plan0059_shadow_integrity", "Inherited P1 shadow failed quick_check.")

    activation_manifest_path = (
        plan0059_runtime_root.expanduser().absolute()
        / f"activation-{plan0059_activation_sha256[:24]}"
        / "private-manifest.json"
    )
    require_private_file(activation_manifest_path, plan0059_runtime_root)
    activation_manifest = read_private_object(activation_manifest_path)
    cohort = (activation_manifest.get("cohort") or {}).get("members") or []
    cases = [
        _source_case(active_database, str(item.get("document_id") or ""))
        for item in cohort
    ]
    if len(cases) != EXPECTED_RECORDINGS or sum(
        len(case["speaker_refs"]) for case in cases
    ) != EXPECTED_SPEAKERS:
        _fail("plan0060_denominator_drift", "Plan 0060 inherited denominator drifted.")

    authority_root = plan0057_authority_manifest.parents[2]
    require_private_file(plan0057_authority_manifest, authority_root)
    prior = read_private_object(plan0057_authority_manifest)
    authority = prior.get("preview") if isinstance(prior.get("preview"), Mapping) else {}
    if (
        set(authority.get("allowlisted_subject_ids") or [])
        != set(acoustic_plan0057.ALLOWLISTED_SUBJECT_IDS)
        or (authority.get("local_runtime") or {}).get("network_required") is not False
        or sha256_file(plan0057_authority_manifest)
        != (plan0060["manifest"].get("acoustic") or {}).get(
            "authority_manifest_sha256"
        )
    ):
        _fail("plan0060_acoustic_authority_invalid", "Acoustic authority drifted.")

    identity_state = acoustic_plan0057._current_identity_state()
    if identity_state.get("snapshot_sha256") != (
        plan0060["manifest"].get("live") or {}
    ).get("identity_state_sha256"):
        _fail("plan0060_identity_state_drift", "Identity/profile/reference state drifted.")
    return {
        "plan0060": plan0060,
        "activation": activation,
        "p1": p1,
        "p1_manifest": p1_manifest,
        "active_shadow_root": active_shadow_root,
        "active_database": active_database,
        "cases": cases,
        "authority": authority,
        "identity_state": identity_state,
    }


def execute_acoustic_lane(
    *,
    runtime_root: Path,
    activation_sha256: str,
    plan0059_runtime_root: Path,
    plan0059_activation_sha256: str,
    plan0057_authority_manifest: Path,
) -> dict[str, Any]:
    paths = _lane_paths(runtime_root, activation_sha256, "p2a-acoustic")
    if paths["receipt"].exists():
        return replay_acoustic_lane(
            runtime_root=runtime_root,
            activation_sha256=activation_sha256,
        )
    if paths["run"].exists():
        _fail("incomplete_plan0060_p2a", "P2A directory exists without a receipt.")
    shared = _shared_inputs(
        runtime_root=runtime_root,
        activation_sha256=activation_sha256,
        plan0059_runtime_root=plan0059_runtime_root,
        plan0059_activation_sha256=plan0059_activation_sha256,
        plan0057_authority_manifest=plan0057_authority_manifest,
    )
    ensure_private_tree(paths["root"], paths["run"])
    ensure_private_tree(paths["root"], paths["sources"])
    created_at = str(shared["plan0060"]["manifest"]["activated_at"])
    identity_before = shared["identity_state"]
    results = []
    for ordinal, case in enumerate(shared["cases"], start=1):
        source_root = paths["sources"] / (
            f"{ordinal:02d}-{hashlib.sha256(case['document_id'].encode()).hexdigest()[:16]}"
        )
        ensure_private_tree(paths["root"], source_root)
        review = _acoustic_review_for_case(
            case,
            authority=shared["authority"],
            source_root=source_root,
            identity_state=identity_before,
            created_at=created_at,
        )
        bundle = adapt_acoustic_review(
            review,
            conversation_id=case["conversation_id"],
            recording_id=case["recording_id"],
            document_id=case["document_id"],
            transcript_sha256=case["transcript_sha256"],
            model_versions=_model_versions(shared["authority"]),
            created_at=created_at,
        )
        results.append(
            {
                "document_id": case["document_id"],
                "recording_id": case["recording_id"],
                "speaker_ref_count": len(case["speaker_refs"]),
                "execution_content_sha256": review["execution_content_sha256"],
                "bundle_id": bundle.bundle_id,
                "bundle": asdict(bundle),
            }
        )
    identity_after = acoustic_plan0057._current_identity_state()
    if identity_after != identity_before:
        _fail("plan0060_identity_state_mutation", "P2A changed identity state.")
    manifest = {
        "schema_version": P2A_MANIFEST_VERSION,
        "status": "acoustic_lane_complete",
        "activation_sha256": activation_sha256,
        "created_at": created_at,
        "recording_count": len(results),
        "speaker_ref_count": sum(item["speaker_ref_count"] for item in results),
        "identity_state_before": identity_before,
        "identity_state_after": identity_after,
        "results": results,
        "negative_actions": negative_action_vector(),
    }
    if (
        manifest["recording_count"] != EXPECTED_RECORDINGS
        or manifest["speaker_ref_count"] != EXPECTED_SPEAKERS
    ):
        _fail("plan0060_p2a_incomplete", "P2A denominator is incomplete.")
    ensure_private_tree(paths["root"], paths["run"])
    write_immutable_private_json(paths["manifest"], manifest)
    receipt = {
        "schema_version": P2A_RECEIPT_VERSION,
        "status": "acoustic_lane_complete",
        "content_sha256": canonical_artifact_hash(manifest),
        "manifest_sha256": sha256_file(paths["manifest"]),
        "recording_count": EXPECTED_RECORDINGS,
        "speaker_ref_count": EXPECTED_SPEAKERS,
        "bundle_count": EXPECTED_RECORDINGS,
        "identity_state_unchanged": True,
        "live_mutation_count": 0,
        "negative_actions_preserved": True,
    }
    write_immutable_private_json(paths["receipt"], receipt)
    return {**receipt, "manifest_path": str(paths["manifest"]), "idempotent_replay": False}


def replay_acoustic_lane(
    *, runtime_root: Path, activation_sha256: str
) -> dict[str, Any]:
    paths = _lane_paths(runtime_root, activation_sha256, "p2a-acoustic")
    manifest, receipt = _read_lane(paths)
    if (
        manifest.get("schema_version") != P2A_MANIFEST_VERSION
        or manifest.get("activation_sha256") != activation_sha256
        or manifest.get("recording_count") != EXPECTED_RECORDINGS
        or manifest.get("speaker_ref_count") != EXPECTED_SPEAKERS
        or receipt.get("bundle_count") != EXPECTED_RECORDINGS
        or manifest.get("identity_state_before") != manifest.get("identity_state_after")
    ):
        _fail("plan0060_p2a_replay_invalid", "P2A receipt binding is invalid.")
    return {**receipt, "manifest_path": str(paths["manifest"]), "idempotent_replay": True}


def execute_context_lane(
    *,
    runtime_root: Path,
    state_root: Path,
    activation_sha256: str,
    plan0059_runtime_root: Path,
    plan0059_activation_sha256: str,
    plan0057_authority_manifest: Path,
) -> dict[str, Any]:
    paths = _lane_paths(runtime_root, activation_sha256, "p2b-context")
    if paths["receipt"].exists():
        return replay_context_lane(
            runtime_root=runtime_root,
            activation_sha256=activation_sha256,
        )
    if paths["run"].exists():
        _fail("incomplete_plan0060_p2b", "P2B directory exists without a receipt.")
    shared = _shared_inputs(
        runtime_root=runtime_root,
        activation_sha256=activation_sha256,
        plan0059_runtime_root=plan0059_runtime_root,
        plan0059_activation_sha256=plan0059_activation_sha256,
        plan0057_authority_manifest=plan0057_authority_manifest,
    )
    ensure_private_tree(paths["root"], paths["run"])
    ensure_private_tree(paths["root"], paths["context_root"])
    _sqlite_backup(shared["active_database"], paths["context_database"])
    if _quick_check(paths["context_database"]) != "ok":
        _fail("plan0060_context_shadow_integrity", "P2B copy failed quick_check.")
    contacts = _contact_rows(paths["context_database"])
    resolved = normalize_explicit_provider_scopes(
        provenance_config.speaker_preprocessing_source_configs_from_provenance(
            state_root=state_root
        )
    )
    expected_scope = shared["plan0060"]["manifest"].get("context") or {}
    scope_hash = hashlib.sha256(
        (
            json.dumps(
                resolved.get("retrieval_sources") or [],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
    ).hexdigest()
    if (
        len(resolved.get("retrieval_sources") or [])
        != int(expected_scope.get("provider_scope_count") or -1)
        or scope_hash != expected_scope.get("provider_scope_sha256")
    ):
        _fail("plan0060_provider_scope_drift", "P2B provider scopes drifted.")

    p1_manifest = shared["p1_manifest"]
    reconciliation = p1_manifest.get("reconciliation_preview") or {}
    watermark = str((p1_manifest.get("active_shadow") or {}).get("table_counts_sha256") or "")
    created_at = str(shared["plan0060"]["manifest"]["activated_at"])
    run_id = _stable_id("plan0060-p2b", activation_sha256)
    results = []
    for case in shared["cases"]:
        context_bundle, candidate_snapshot, private_context = _context_for_case(
            case,
            active_shadow_root=paths["context_root"],
            resolved=resolved,
            contacts=contacts,
            reconciliation=reconciliation,
            projection_watermark=watermark,
            created_at=created_at,
            run_id=run_id,
        )
        results.append(
            {
                "document_id": case["document_id"],
                "recording_id": case["recording_id"],
                "speaker_ref_count": len(case["speaker_refs"]),
                "context_bundle_id": context_bundle.bundle_id,
                "context_bundle": asdict(context_bundle),
                "candidate_snapshot_id": candidate_snapshot.snapshot_id,
                "candidate_snapshot": asdict(candidate_snapshot),
                "private_context": private_context,
            }
        )
    identity_after = acoustic_plan0057._current_identity_state()
    if identity_after != shared["identity_state"]:
        _fail("plan0060_identity_state_mutation", "P2B changed identity state.")
    manifest = {
        "schema_version": P2B_MANIFEST_VERSION,
        "status": "context_lane_complete",
        "activation_sha256": activation_sha256,
        "created_at": created_at,
        "run_id": run_id,
        "recording_count": len(results),
        "speaker_ref_count": sum(item["speaker_ref_count"] for item in results),
        "provider_scope_count": len(resolved.get("retrieval_sources") or []),
        "provider_scope_sha256": scope_hash,
        "context_shadow_sha256": sha256_file(paths["context_database"]),
        "context_shadow_quick_check": _quick_check(paths["context_database"]),
        "identity_state_before": shared["identity_state"],
        "identity_state_after": identity_after,
        "results": results,
        "negative_actions": negative_action_vector(),
    }
    if (
        manifest["recording_count"] != EXPECTED_RECORDINGS
        or manifest["speaker_ref_count"] != EXPECTED_SPEAKERS
    ):
        _fail("plan0060_p2b_incomplete", "P2B denominator is incomplete.")
    write_immutable_private_json(paths["manifest"], manifest)
    receipt = {
        "schema_version": P2B_RECEIPT_VERSION,
        "status": "context_lane_complete",
        "content_sha256": canonical_artifact_hash(manifest),
        "manifest_sha256": sha256_file(paths["manifest"]),
        "recording_count": EXPECTED_RECORDINGS,
        "speaker_ref_count": EXPECTED_SPEAKERS,
        "bundle_count": EXPECTED_RECORDINGS,
        "candidate_snapshot_count": EXPECTED_RECORDINGS,
        "provider_scope_count": manifest["provider_scope_count"],
        "provider_failure_count": sum(
            int(item["private_context"]["provider_failure_count"]) for item in results
        ),
        "included_evidence_count": sum(
            int(item["private_context"]["included_evidence_count"]) for item in results
        ),
        "identity_state_unchanged": True,
        "live_mutation_count": 0,
        "negative_actions_preserved": True,
    }
    write_immutable_private_json(paths["receipt"], receipt)
    return {**receipt, "manifest_path": str(paths["manifest"]), "idempotent_replay": False}


def replay_context_lane(
    *, runtime_root: Path, activation_sha256: str
) -> dict[str, Any]:
    paths = _lane_paths(runtime_root, activation_sha256, "p2b-context")
    manifest, receipt = _read_lane(paths)
    if (
        manifest.get("schema_version") != P2B_MANIFEST_VERSION
        or manifest.get("activation_sha256") != activation_sha256
        or manifest.get("recording_count") != EXPECTED_RECORDINGS
        or manifest.get("speaker_ref_count") != EXPECTED_SPEAKERS
        or receipt.get("bundle_count") != EXPECTED_RECORDINGS
        or receipt.get("candidate_snapshot_count") != EXPECTED_RECORDINGS
        or manifest.get("context_shadow_sha256") != sha256_file(paths["context_database"])
        or manifest.get("context_shadow_quick_check") != "ok"
        or _quick_check(paths["context_database"]) != "ok"
        or manifest.get("identity_state_before") != manifest.get("identity_state_after")
    ):
        _fail("plan0060_p2b_replay_invalid", "P2B receipt binding is invalid.")
    return {**receipt, "manifest_path": str(paths["manifest"]), "idempotent_replay": True}


def _read_lane(paths: Mapping[str, Path]) -> tuple[dict[str, Any], dict[str, Any]]:
    require_private_file(paths["manifest"], paths["root"])
    require_private_file(paths["receipt"], paths["root"])
    manifest = read_private_object(paths["manifest"])
    receipt = read_private_object(paths["receipt"])
    if (
        receipt.get("content_sha256") != canonical_artifact_hash(manifest)
        or receipt.get("manifest_sha256") != sha256_file(paths["manifest"])
        or receipt.get("negative_actions_preserved") is not True
        or receipt.get("live_mutation_count") != 0
        or any((manifest.get("negative_actions") or {}).values())
    ):
        _fail("plan0060_lane_replay_invalid", "Plan 0060 lane receipt is invalid.")
    return manifest, receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Execute Plan 0060 independent evidence lanes.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("execute-acoustic", "execute-context"):
        lane = subparsers.add_parser(command)
        lane.add_argument("--runtime-root", type=Path, required=True)
        lane.add_argument("--activation-sha256", required=True)
        lane.add_argument("--plan0059-runtime-root", type=Path, required=True)
        lane.add_argument("--plan0059-activation-sha256", required=True)
        lane.add_argument("--plan0057-authority-manifest", type=Path, required=True)
        if command == "execute-context":
            lane.add_argument("--state-root", type=Path, required=True)
    for command in ("replay-acoustic", "replay-context"):
        replay = subparsers.add_parser(command)
        replay.add_argument("--runtime-root", type=Path, required=True)
        replay.add_argument("--activation-sha256", required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "execute-acoustic":
            result = execute_acoustic_lane(
                runtime_root=args.runtime_root,
                activation_sha256=args.activation_sha256,
                plan0059_runtime_root=args.plan0059_runtime_root,
                plan0059_activation_sha256=args.plan0059_activation_sha256,
                plan0057_authority_manifest=args.plan0057_authority_manifest,
            )
        elif args.command == "execute-context":
            result = execute_context_lane(
                runtime_root=args.runtime_root,
                state_root=args.state_root,
                activation_sha256=args.activation_sha256,
                plan0059_runtime_root=args.plan0059_runtime_root,
                plan0059_activation_sha256=args.plan0059_activation_sha256,
                plan0057_authority_manifest=args.plan0057_authority_manifest,
            )
        elif args.command == "replay-acoustic":
            result = replay_acoustic_lane(
                runtime_root=args.runtime_root,
                activation_sha256=args.activation_sha256,
            )
        else:
            result = replay_context_lane(
                runtime_root=args.runtime_root,
                activation_sha256=args.activation_sha256,
            )
    except IdentityOrchestrationError as exc:
        print(json.dumps({"status": "error", "reason_code": exc.reason_code, "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
