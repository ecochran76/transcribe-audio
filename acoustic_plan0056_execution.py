from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from collections.abc import Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import acoustic_plan0056_pilot as pilot
import acoustic_generation5_e2 as generation5_e2
from acoustic_audio_derivatives import (
    ensure_private_tree,
    read_private_object,
    require_private_file,
    write_immutable_private_json,
)


PROPOSAL_EVIDENCE_SCHEMA = "transcribe-audio.plan0056-pilot-proposal-evidence.v1"
EXPECTED_UNIT_COUNT = 9
ASSIGNMENT_MINIMUM_SUPPORTING_UNITS = 6
ASSIGNMENT_MINIMUM_CANDIDATE_FAMILIES = 2
ASSIGNMENT_MAXIMUM_OPPOSING_UNITS = 0
P0_CONTENT_SHA256 = "7477fed61e2e2b8035523a91a0afd763306493423d6ddeebfa96e274d9a9522d"
P0_MANIFEST_SHA256 = "a7e195e6e9efeaff85ec64e17a1eb30d0dbada50758505ebff089782cd064a80"
EXECUTION_AUTHORITY_SCHEMA = "transcribe-audio.plan0056-execution-authority.v1"
EXECUTION_AUTHORITY_MANIFEST_SCHEMA = (
    "transcribe-audio.plan0056-execution-authority-manifest.v1"
)
EXECUTION_AUTHORITY_RECEIPT_SCHEMA = (
    "transcribe-audio.plan0056-execution-authority-receipt.v1"
)
EXECUTION_AUTHORITY_REPLAY_SCHEMA = (
    "transcribe-audio.plan0056-execution-authority-replay.v1"
)
PLAN_PATH = Path(
    "docs/dev/plans/0056-2026-08-05-enrolled-only-acoustic-pilot-identity-guard.md"
)
MODULE_PATH = Path(__file__).name
RUNNER_PATH = Path("acoustic_plan0056_runner.py")
DEFAULT_RUNTIME_ROOT = Path("~/.local/state/transcribe-audio/plan-0056/p1")
DEFAULT_DIARIZATION_ROOT = Path(
    "~/.local/state/transcribe-audio/plan-0037/speech-preparation/acquisitions/"
    "pyannote-community-1-20260731"
)
DEFAULT_WHISPER_CACHE_ROOT = Path(
    "~/.cache/huggingface/hub/models--Systran--faster-whisper-small.en"
)
SHA256_RE = re.compile(r"[a-f0-9]{64}")


class Plan0056ExecutionError(ValueError):
    """Raised when local pilot execution cannot remain bounded and replayable."""


def _canonical_hash(value: Any) -> str:
    body = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(arguments: Sequence[str], *, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", *arguments],
        cwd=Path(__file__).resolve().parent,
        capture_output=True,
        text=not binary,
        check=False,
    )
    if result.returncode:
        raise Plan0056ExecutionError("Repository authority is unavailable.")
    return result.stdout if binary else result.stdout.strip()


def _repository_authority() -> dict[str, Any]:
    if _git(["status", "--porcelain=v1", "--untracked-files=normal"]):
        raise Plan0056ExecutionError("Repository must be clean.")
    divergence = str(
        _git(["rev-list", "--left-right", "--count", "@{upstream}...HEAD"])
    ).split()
    if divergence != ["0", "0"]:
        raise Plan0056ExecutionError("Repository must be upstream-even.")
    commit = str(_git(["rev-parse", "HEAD"]))
    hashes: dict[str, str] = {}
    for relative in (MODULE_PATH, RUNNER_PATH.as_posix(), PLAN_PATH.as_posix()):
        committed = _git(["show", f"{commit}:{relative}"], binary=True)
        if not isinstance(committed, bytes):
            raise Plan0056ExecutionError("Committed execution authority is unavailable.")
        current = Path(__file__).resolve().parent / relative
        digest = hashlib.sha256(committed).hexdigest()
        if digest != _sha256_file(current):
            raise Plan0056ExecutionError("Committed execution authority drifted.")
        hashes[relative] = digest
    return {
        "commit": commit,
        "authority_file_sha256": hashes,
        "clean": True,
        "upstream_ahead": 0,
        "upstream_behind": 0,
    }


def _file_inventory(root: Path) -> dict[str, Any]:
    absolute = root.expanduser().absolute()
    if not absolute.is_dir() or absolute.is_symlink():
        raise Plan0056ExecutionError("A required local model root is unavailable.")
    rows = []
    for path in sorted(absolute.rglob("*")):
        if path.is_symlink():
            resolved = path.resolve()
            if not resolved.is_file():
                raise Plan0056ExecutionError("A local model symlink is invalid.")
            target = resolved
        elif path.is_file():
            target = path
        else:
            continue
        rows.append(
            {
                "relative_path": path.relative_to(absolute).as_posix(),
                "size_bytes": target.stat().st_size,
                "sha256": _sha256_file(target),
            }
        )
    if not rows:
        raise Plan0056ExecutionError("A required local model inventory is empty.")
    return {
        "root": str(absolute),
        "file_count": len(rows),
        "file_set_sha256": _canonical_hash(rows),
    }


def local_runtime_inventory(
    *,
    diarization_root: Path = DEFAULT_DIARIZATION_ROOT,
    whisper_cache_root: Path = DEFAULT_WHISPER_CACHE_ROOT,
) -> dict[str, Any]:
    import torch

    distributions = {}
    for name in ("pyannote.audio", "faster-whisper", "torch", "torchaudio"):
        try:
            distributions[name] = version(name)
        except PackageNotFoundError as exc:
            raise Plan0056ExecutionError("The local speech runtime is incomplete.") from exc
    diarization = _file_inventory(diarization_root)
    snapshots = sorted(
        path
        for path in (whisper_cache_root.expanduser().absolute() / "snapshots").iterdir()
        if path.is_dir() and not path.is_symlink()
    )
    if len(snapshots) != 1:
        raise Plan0056ExecutionError("The local whisper snapshot is ambiguous.")
    transcription = _file_inventory(snapshots[0])
    core = {
        "schema_version": "transcribe-audio.plan0056-local-runtime.v1",
        "installed_distributions": distributions,
        "diarization_model": diarization,
        "transcription_model": transcription,
        "network_required": False,
        "diarization_model_local": True,
        "transcription_model_local": True,
        "compute_device": "cuda" if torch.cuda.is_available() else "cpu",
        "compute_device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        ),
    }
    return {**core, "runtime_sha256": _canonical_hash(core)}


def proposals_from_matrices(
    matrices: Sequence[Mapping[str, Any]],
    *,
    expected_speaker_refs: Sequence[str],
    allowlisted_subject_ids: Sequence[str],
) -> dict[str, Any]:
    """Apply the frozen no-opposition consensus rule to nine acoustic units."""

    refs = tuple(str(item) for item in expected_speaker_refs)
    subjects = tuple(str(item) for item in allowlisted_subject_ids)
    if len(matrices) != EXPECTED_UNIT_COUNT:
        raise Plan0056ExecutionError("Exactly nine acoustic matrices are required.")
    support: dict[str, dict[str, list[tuple[str, str]]]] = {
        ref: {subject: [] for subject in subjects} for ref in refs
    }
    observed_units: set[tuple[str, str]] = set()
    for raw_matrix in matrices:
        candidate_id = str(raw_matrix.get("candidate_id") or "")
        method_id = str(raw_matrix.get("method_id") or "")
        unit = (candidate_id, method_id)
        if not all(unit) or unit in observed_units:
            raise Plan0056ExecutionError("Acoustic matrix units are invalid or duplicated.")
        observed_units.add(unit)
        try:
            threshold = float(raw_matrix["threshold"])
        except (KeyError, TypeError, ValueError) as exc:
            raise Plan0056ExecutionError("Acoustic matrix threshold is invalid.") from exc
        rows = raw_matrix.get("rows")
        if not isinstance(rows, list) or len(rows) != len(refs):
            raise Plan0056ExecutionError("Acoustic matrix speaker denominator is incomplete.")
        by_ref: dict[str, Mapping[str, Any]] = {}
        for row in rows:
            if not isinstance(row, Mapping):
                raise Plan0056ExecutionError("Acoustic matrix row is invalid.")
            speaker_ref = str(row.get("speaker_ref") or "")
            if speaker_ref not in refs or speaker_ref in by_ref:
                raise Plan0056ExecutionError("Acoustic matrix speaker references are invalid.")
            by_ref[speaker_ref] = row
        for speaker_ref in refs:
            scores = by_ref[speaker_ref].get("scores")
            if not isinstance(scores, list) or len(scores) != 2:
                raise Plan0056ExecutionError("Each acoustic row requires two scores.")
            parsed: list[tuple[str, float]] = []
            for score in scores:
                if not isinstance(score, Mapping):
                    raise Plan0056ExecutionError("An acoustic score is invalid.")
                subject_id = str(score.get("subject_id") or "")
                try:
                    value = float(score["score"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise Plan0056ExecutionError("An acoustic score is invalid.") from exc
                if subject_id not in subjects or not math.isfinite(value):
                    raise Plan0056ExecutionError("An acoustic score is unbound or nonfinite.")
                parsed.append((subject_id, value))
            if {item[0] for item in parsed} != set(subjects):
                raise Plan0056ExecutionError("Acoustic score subjects are incomplete.")
            ranked = sorted(parsed, key=lambda item: (item[1], item[0]), reverse=True)
            if ranked[0][1] >= threshold and ranked[1][1] < threshold:
                support[speaker_ref][ranked[0][0]].append(unit)

    core_proposals: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    for speaker_ref in refs:
        ranked_subjects = sorted(
            subjects,
            key=lambda subject: (len(support[speaker_ref][subject]), subject),
            reverse=True,
        )
        winner, other = ranked_subjects
        winner_units = support[speaker_ref][winner]
        other_units = support[speaker_ref][other]
        family_count = len({candidate for candidate, _method in winner_units})
        if (
            len(winner_units) >= ASSIGNMENT_MINIMUM_SUPPORTING_UNITS
            and family_count >= ASSIGNMENT_MINIMUM_CANDIDATE_FAMILIES
            and len(other_units) <= ASSIGNMENT_MAXIMUM_OPPOSING_UNITS
        ):
            disposition = "assign"
            subject_id: str | None = winner
            confidence_band = "high" if len(winner_units) == EXPECTED_UNIT_COUNT else "medium"
        elif winner_units or other_units:
            disposition = "review"
            subject_id = winner
            confidence_band = "low"
        else:
            disposition = "abstain"
            subject_id = None
            confidence_band = "none"
        core_proposals.append(
            {
                "speaker_ref": speaker_ref,
                "disposition": disposition,
                "subject_id": subject_id,
                "confidence_band": confidence_band,
                "rationale": (
                    f"Frozen consensus: {len(winner_units)} supporting units across "
                    f"{family_count} model families and {len(other_units)} opposing units."
                ),
            }
        )
        evidence.append(
            {
                **core_proposals[-1],
                "supporting_unit_count": len(winner_units),
                "supporting_candidate_family_count": family_count,
                "opposing_unit_count": len(other_units),
                "supporting_units": [list(item) for item in sorted(winner_units)],
                "opposing_units": [list(item) for item in sorted(other_units)],
            }
        )
    validated = pilot.validate_pilot_proposals(
        {"proposals": core_proposals},
        expected_speaker_refs=refs,
        allowlisted_subject_ids=subjects,
    )
    core = {
        "schema_version": PROPOSAL_EVIDENCE_SCHEMA,
        "speaker_count": len(refs),
        "allowlisted_subject_ids": list(subjects),
        "validated_proposals_sha256": validated["content_sha256"],
        "proposals": evidence,
        "requires_human_review": True,
        "will_apply_assignments": False,
    }
    return {**core, "content_sha256": _canonical_hash(core)}


def preview_plan0056_execution(
    *,
    p0_authority: Mapping[str, Any],
    repository_authority: Mapping[str, Any],
    local_runtime: Mapping[str, Any],
    threshold_units: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Freeze the exact local-only actions allowed after P0 and before decode."""

    if (
        p0_authority.get("preview_content_sha256") != P0_CONTENT_SHA256
        or p0_authority.get("manifest_sha256") != P0_MANIFEST_SHA256
        or p0_authority.get("idempotent_replay") is not True
        or p0_authority.get("source_count") != 1
        or len(p0_authority.get("allowlisted_subject_ids") or []) != 2
    ):
        raise Plan0056ExecutionError("Frozen P0 authority is invalid or drifted.")
    if (
        repository_authority.get("clean") is not True
        or repository_authority.get("upstream_ahead") != 0
        or repository_authority.get("upstream_behind") != 0
    ):
        raise Plan0056ExecutionError("Repository must be clean and upstream-even.")
    if (
        local_runtime.get("network_required") is not False
        or local_runtime.get("diarization_model_local") is not True
        or local_runtime.get("transcription_model_local") is not True
        or local_runtime.get("compute_device") not in {"cpu", "cuda"}
        or not local_runtime.get("runtime_sha256")
    ):
        raise Plan0056ExecutionError("The local speech runtime is incomplete.")
    units = [dict(item) for item in threshold_units]
    observed = {
        (str(item.get("candidate_id") or ""), str(item.get("method_id") or ""))
        for item in units
    }
    if len(units) != EXPECTED_UNIT_COUNT or len(observed) != EXPECTED_UNIT_COUNT:
        raise Plan0056ExecutionError("Exactly nine threshold units are required.")

    core = {
        "schema_version": EXECUTION_AUTHORITY_SCHEMA,
        "status": "ready_to_freeze_before_local_execution",
        "p0_authority": dict(p0_authority),
        "repository_authority": dict(repository_authority),
        "local_runtime": dict(local_runtime),
        "threshold_units": units,
        "threshold_unit_set_sha256": _canonical_hash(units),
        "diarization_policy": {
            "minimum_speakers": 1,
            "maximum_speakers": 6,
            "model_output_labels_are_not_identities": True,
        },
        "review_clip_policy": {
            "minimum_turn_seconds": 2.0,
            "maximum_turn_seconds": 8.0,
            "maximum_turns_per_speaker": 6,
            "target_seconds_per_speaker": 24.0,
            "minimum_usable_seconds_per_speaker": 6.0,
            "sample_rate": 16_000,
            "channels": 1,
        },
        "consensus_policy": {
            "assignment_minimum_supporting_units": ASSIGNMENT_MINIMUM_SUPPORTING_UNITS,
            "assignment_minimum_candidate_families": ASSIGNMENT_MINIMUM_CANDIDATE_FAMILIES,
            "assignment_maximum_opposing_units": ASSIGNMENT_MAXIMUM_OPPOSING_UNITS,
            "high_confidence_supporting_units": EXPECTED_UNIT_COUNT,
            "human_confirmation_required": True,
        },
        "action_vector": {
            "decode_source_to_private_pcm": True,
            "run_local_diarization": True,
            "create_private_review_clips": True,
            "transcribe_review_clips_locally": True,
            "run_nine_acoustic_matrices": True,
            "prepare_subject_id_proposals": True,
            "prepare_human_review": True,
            "read_pilot_outcome_gold": False,
            "create_or_mutate_identity_records": False,
            "mutate_profiles_or_references": False,
            "write_external_provider": False,
            "apply_speaker_assignments": False,
            "enable_default_integration": False,
            "run_historical_reprocessing": False,
        },
        "contains_pilot_outcome_gold": False,
        "contains_display_names": False,
    }
    return {**core, "content_sha256": _canonical_hash(core)}


def _authority_paths(runtime_root: Path, content_sha256: str) -> dict[str, Path]:
    root = runtime_root.expanduser().absolute()
    run = root / f"execution-authority-{content_sha256[:24]}"
    return {
        "root": root,
        "run": run,
        "manifest": run / "private-manifest.json",
        "receipt": run / "receipt.json",
    }


def _authority_receipt(
    preview: Mapping[str, Any], manifest_sha256: str
) -> dict[str, Any]:
    return {
        "schema_version": EXECUTION_AUTHORITY_RECEIPT_SCHEMA,
        "status": "frozen_before_local_execution",
        "preview_content_sha256": preview["content_sha256"],
        "manifest_sha256": manifest_sha256,
        "p0_content_sha256": preview["p0_authority"]["preview_content_sha256"],
        "source_set_sha256": preview["p0_authority"]["source_set_sha256"],
        "allowlisted_subject_ids": preview["p0_authority"]["allowlisted_subject_ids"],
        "local_runtime_sha256": preview["local_runtime"]["runtime_sha256"],
        "threshold_unit_set_sha256": preview["threshold_unit_set_sha256"],
        "action_vector": preview["action_vector"],
        "mode": "0600",
        "did_decode_audio": False,
        "did_run_diarization": False,
        "did_run_acoustic_models": False,
        "did_prepare_proposals": False,
        "did_mutate_identity_or_profile_state": False,
        "did_write_external_provider": False,
    }


def freeze_plan0056_execution_authority(
    reviewed_preview: Mapping[str, Any],
    *,
    expected_content_sha256: str,
    runtime_root: Path,
) -> dict[str, Any]:
    preview = dict(reviewed_preview)
    core = {key: value for key, value in preview.items() if key != "content_sha256"}
    if (
        preview.get("schema_version") != EXECUTION_AUTHORITY_SCHEMA
        or preview.get("content_sha256") != expected_content_sha256
        or _canonical_hash(core) != expected_content_sha256
        or preview.get("contains_pilot_outcome_gold") is not False
    ):
        raise Plan0056ExecutionError("Reviewed execution authority is stale or unsafe.")
    paths = _authority_paths(runtime_root, expected_content_sha256)
    if paths["receipt"].exists():
        return replay_plan0056_execution_authority(
            expected_content_sha256, runtime_root=runtime_root
        )
    ensure_private_tree(paths["root"], paths["run"])
    manifest = {
        "schema_version": EXECUTION_AUTHORITY_MANIFEST_SCHEMA,
        "status": "frozen_before_local_execution",
        "preview": preview,
    }
    write_immutable_private_json(paths["manifest"], manifest)
    receipt = _authority_receipt(preview, _sha256_file(paths["manifest"]))
    write_immutable_private_json(paths["receipt"], receipt)
    return {**receipt, "idempotent_replay": False}


def replay_plan0056_execution_authority(
    expected_content_sha256: str,
    *,
    runtime_root: Path,
) -> dict[str, Any]:
    paths = _authority_paths(runtime_root, expected_content_sha256)
    try:
        require_private_file(paths["manifest"], paths["root"])
        require_private_file(paths["receipt"], paths["root"])
        manifest = read_private_object(paths["manifest"])
        receipt = read_private_object(paths["receipt"])
    except (OSError, ValueError) as exc:
        raise Plan0056ExecutionError("Frozen execution authority is unavailable.") from exc
    preview = manifest.get("preview")
    if not isinstance(preview, Mapping):
        raise Plan0056ExecutionError("Frozen execution authority is invalid.")
    core = {key: value for key, value in preview.items() if key != "content_sha256"}
    expected_manifest = {
        "schema_version": EXECUTION_AUTHORITY_MANIFEST_SCHEMA,
        "status": "frozen_before_local_execution",
        "preview": dict(preview),
    }
    expected_receipt = _authority_receipt(preview, _sha256_file(paths["manifest"]))
    if (
        manifest != expected_manifest
        or receipt != expected_receipt
        or preview.get("content_sha256") != expected_content_sha256
        or _canonical_hash(core) != expected_content_sha256
    ):
        raise Plan0056ExecutionError("Frozen execution authority drifted.")
    return {
        **receipt,
        "replay_schema_version": EXECUTION_AUTHORITY_REPLAY_SCHEMA,
        "idempotent_replay": True,
    }


def build_live_execution_authority() -> dict[str, Any]:
    p0 = pilot.replay_plan0056_authority(
        P0_CONTENT_SHA256, runtime_root=pilot.DEFAULT_RUNTIME_ROOT
    )
    thresholds = generation5_e2._threshold_authority()
    units = [
        {
            key: item[key]
            for key in ("candidate_id", "method_id", "threshold", "temperature")
        }
        for item in thresholds["thresholds"]
    ]
    return preview_plan0056_execution(
        p0_authority=p0,
        repository_authority=_repository_authority(),
        local_runtime=local_runtime_inventory(),
        threshold_units=units,
    )


def portable_execution_authority(preview: Mapping[str, Any]) -> dict[str, Any]:
    runtime = preview.get("local_runtime", {})
    return {
        "schema_version": preview.get("schema_version"),
        "status": preview.get("status"),
        "content_sha256": preview.get("content_sha256"),
        "p0_content_sha256": preview.get("p0_authority", {}).get(
            "preview_content_sha256"
        ),
        "source_set_sha256": preview.get("p0_authority", {}).get(
            "source_set_sha256"
        ),
        "allowlisted_subject_ids": preview.get("p0_authority", {}).get(
            "allowlisted_subject_ids"
        ),
        "repository_authority": preview.get("repository_authority"),
        "local_runtime": {
            "schema_version": runtime.get("schema_version"),
            "runtime_sha256": runtime.get("runtime_sha256"),
            "installed_distributions": runtime.get("installed_distributions"),
            "diarization_model_file_count": runtime.get("diarization_model", {}).get(
                "file_count"
            ),
            "diarization_model_file_set_sha256": runtime.get(
                "diarization_model", {}
            ).get("file_set_sha256"),
            "transcription_model_file_count": runtime.get(
                "transcription_model", {}
            ).get("file_count"),
            "transcription_model_file_set_sha256": runtime.get(
                "transcription_model", {}
            ).get("file_set_sha256"),
            "network_required": runtime.get("network_required"),
            "compute_device": runtime.get("compute_device"),
            "compute_device_name": runtime.get("compute_device_name"),
        },
        "threshold_unit_set_sha256": preview.get("threshold_unit_set_sha256"),
        "diarization_policy": preview.get("diarization_policy"),
        "review_clip_policy": preview.get("review_clip_policy"),
        "consensus_policy": preview.get("consensus_policy"),
        "action_vector": preview.get("action_vector"),
        "contains_pilot_outcome_gold": preview.get("contains_pilot_outcome_gold"),
        "contains_private_paths": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze and replay the local Plan 0056 execution authority."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preview")
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--expected-content-sha256", required=True)
    freeze.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    replay = subparsers.add_parser("replay")
    replay.add_argument("--content-sha256", required=True)
    replay.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "replay":
        if not SHA256_RE.fullmatch(args.content_sha256):
            raise Plan0056ExecutionError("Execution authority hash is invalid.")
        result = replay_plan0056_execution_authority(
            args.content_sha256, runtime_root=args.runtime_root
        )
    else:
        preview = build_live_execution_authority()
        if args.command == "preview":
            result = portable_execution_authority(preview)
        else:
            if args.expected_content_sha256 != preview["content_sha256"]:
                raise Plan0056ExecutionError("Reviewed execution authority hash is stale.")
            result = freeze_plan0056_execution_authority(
                preview,
                expected_content_sha256=args.expected_content_sha256,
                runtime_root=args.runtime_root,
            )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
