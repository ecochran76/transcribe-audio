"""Freeze Plan 0071's structural supplemental development cohort."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import speaker_identity_plan0064_p0 as plan0064_p0
import speaker_identity_plan0071_d0 as d0
import speaker_identity_plan0071_d1 as d1
from acoustic_audio_derivatives import (
    ensure_private_tree,
    read_private_object,
    sha256_file,
    write_immutable_private_json,
)


SCHEMA_VERSION = "transcribe-audio.plan0071-d2-cohort.v1"
RECEIPT_SCHEMA_VERSION = "transcribe-audio.plan0071-d2-cohort-receipt.v1"
DEFAULT_RUNTIME_ROOT = d0.DEFAULT_RUNTIME_ROOT
DEFAULT_TRANSCRIPT_ROOT = plan0064_p0.DEFAULT_TRANSCRIPT_ROOT
D0_ACTIVATION_CONTENT_SHA256 = d1.D0_ACTIVATION_CONTENT_SHA256
D1_RECEIPT_CONTENT_SHA256 = (
    "13c6f879c0297b9fcdc53841954ec320f57b56d4905bcf3f4b379194300863c7"
)
MAX_CONVERSATIONS = 6
REQUIRED_SPEAKER_COUNT = 3
EFFECT_COUNTS = dict(d0.EFFECT_COUNTS)


class Plan0071D2CohortError(ValueError):
    """Raised when the structural supplemental cohort cannot freeze exactly."""


def _hash(value: Any) -> str:
    return d0._hash(value)


def _content(value: Mapping[str, Any]) -> dict[str, Any]:
    return d0._content(value)


def _validate_content(value: Mapping[str, Any], label: str) -> None:
    try:
        d0._validate_content(value, label)
    except d0.Plan0071D0Error as exc:
        raise Plan0071D2CohortError(str(exc)) from exc


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], check=False, capture_output=True, text=True
    )
    if result.returncode:
        raise Plan0071D2CohortError(
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
        raise Plan0071D2CohortError("D2 cohort source authority is unacceptable.")
    return authority


def _artifact_binding(path: Path, root: Path) -> dict[str, Any]:
    selected = path.expanduser().resolve()
    try:
        selected.relative_to(root.expanduser().resolve())
    except ValueError as exc:
        raise Plan0071D2CohortError("A cohort artifact escapes its root.") from exc
    if selected.is_symlink() or not selected.is_file():
        raise Plan0071D2CohortError("A cohort artifact is not a regular file.")
    return {"path": str(selected), "file_sha256": sha256_file(selected)}


def select_structural_units(
    units: Sequence[Mapping[str, Any]],
    *,
    exposed_document_ids: set[str],
    exposed_recording_hashes: set[str],
    limit: int = MAX_CONVERSATIONS,
) -> dict[str, Any]:
    """Select the next-oldest exactly-three-speaker units without identity data."""

    if limit < 1 or limit > MAX_CONVERSATIONS:
        raise Plan0071D2CohortError("Supplemental cohort bound is invalid.")
    considered = []
    selected = []
    seen_hashes: set[str] = set()
    for raw in units:
        unit = dict(raw)
        document_id = str(unit.get("document_id") or "")
        media_sha256 = str(unit.get("source_media_sha256") or "")
        labels = list(unit.get("speaker_labels") or [])
        reasons = []
        if document_id in exposed_document_ids:
            reasons.append("prior_development_document_exposure")
        if media_sha256 in exposed_recording_hashes:
            reasons.append("prior_development_recording_exposure")
        if media_sha256 and media_sha256 in seen_hashes:
            reasons.append("repeated_recording_hash")
        if len(labels) != REQUIRED_SPEAKER_COUNT:
            reasons.append("speaker_count_not_three")
        if not unit.get("recording_time"):
            reasons.append("missing_recording_time")
        if not unit.get("original_recording_filename"):
            reasons.append("missing_original_recording_filename")
        if unit.get("transcript_artifact_valid") is not True:
            reasons.append("transcript_artifact_unavailable")
        if unit.get("source_media_artifact_valid") is not True:
            reasons.append("source_media_unavailable")
        eligible = not reasons
        disposition = "selected_supplemental_development" if eligible else "excluded"
        row = {
            key: value
            for key, value in unit.items()
            if key
            not in {
                "transcript_artifact_valid",
                "source_media_artifact_valid",
            }
        }
        row.update(
            {
                "eligible": eligible,
                "disposition": disposition,
                "reason_codes": sorted(set(reasons)) or ["eligible"],
            }
        )
        considered.append(row)
        if eligible:
            selected.append(row)
        if media_sha256:
            seen_hashes.add(media_sha256)
        if len(selected) == limit:
            break
    if len(selected) != limit:
        raise Plan0071D2CohortError(
            f"Only {len(selected)} of {limit} supplemental recordings are eligible."
        )
    return {
        "considered": considered,
        "selected": selected,
        "considered_count": len(considered),
        "selected_count": len(selected),
        "last_considered_chronological_rank": considered[-1][
            "chronological_rank"
        ],
        "disposition_counts": dict(
            sorted(Counter(item["disposition"] for item in considered).items())
        ),
        "reason_code_counts": dict(
            sorted(
                Counter(
                    reason
                    for item in considered
                    for reason in item["reason_codes"]
                ).items()
            )
        ),
    }


def _candidate_units(transcript_root: Path) -> tuple[list[dict[str, Any]], int]:
    root = transcript_root.expanduser().resolve()
    candidates, total_count = plan0064_p0._candidate_rows(root)
    units = []
    for candidate in candidates:
        row = candidate["row"]
        payload = candidate["payload"]
        utterances = (
            payload.get("utterances")
            if isinstance(payload.get("utterances"), list)
            else []
        )
        labels = sorted(
            {
                str(item.get("speaker") or "").strip()
                for item in utterances
                if isinstance(item, Mapping)
                and str(item.get("speaker") or "").strip()
            }
        )
        transcript_path = Path(str(row.get("stored_path") or ""))
        media_path = Path(str(row.get("media_stored_path") or ""))
        transcript_valid = (
            transcript_path.is_file()
            and not transcript_path.is_symlink()
            and sha256_file(transcript_path) == str(row.get("artifact_sha256") or "")
        )
        media_valid = (
            media_path.is_file()
            and not media_path.is_symlink()
            and sha256_file(media_path) == str(row.get("media_sha256") or "")
        )
        original_path = str(payload.get("source_media_path") or "")
        units.append(
            {
                "chronological_rank": int(candidate["chronological_rank"]),
                "document_id": str(row.get("id") or ""),
                "recording_time": str(candidate.get("recording_time") or ""),
                "recording_id": str(payload.get("recording_id") or ""),
                "conversation_id": str(payload.get("conversation_id") or ""),
                "original_recording_filename": Path(original_path).name,
                "source_media_sha256": str(row.get("media_sha256") or ""),
                "transcript_sha256": str(row.get("artifact_sha256") or ""),
                "speaker_labels": labels,
                "utterance_count": len(utterances),
                "transcript_artifact": (
                    _artifact_binding(transcript_path, root)
                    if transcript_valid
                    else {"path": str(transcript_path), "file_sha256": ""}
                ),
                "source_media_artifact": (
                    _artifact_binding(media_path, root)
                    if media_valid
                    else {"path": str(media_path), "file_sha256": ""}
                ),
                "transcript_artifact_valid": transcript_valid,
                "source_media_artifact_valid": media_valid,
            }
        )
    return units, total_count


def _exposure_authority() -> tuple[dict[str, Any], dict[str, Any]]:
    d0_receipt = d0.replay_activation(runtime_root=DEFAULT_RUNTIME_ROOT)
    if d0_receipt.get("activation_content_sha256") != D0_ACTIVATION_CONTENT_SHA256:
        raise Plan0071D2CohortError("D0 activation authority drifted.")
    manifest = read_private_object(Path(str(d0_receipt["manifest_path"])))
    _validate_content(manifest, "Plan 0071 D0 manifest")
    p65_binding = manifest["artifact_bindings"]["plan0065_d0_manifest"]
    p65_path = Path(str(p65_binding["path"]))
    if sha256_file(p65_path) != p65_binding["file_sha256"]:
        raise Plan0071D2CohortError("Plan 0065 exposure authority drifted.")
    p65 = read_private_object(p65_path)
    _validate_content(p65, "Plan 0065 D0 manifest")
    exposure = p65["plan0064_authority"]["exposure_set"]
    _validate_content(exposure, "Plan 0065 exposure set")
    return manifest, exposure


def build_cohort_manifest(
    *, transcript_root: Path = DEFAULT_TRANSCRIPT_ROOT
) -> dict[str, Any]:
    d1_receipt = d1.replay_d1(runtime_root=DEFAULT_RUNTIME_ROOT)
    if d1_receipt.get("content_sha256") != D1_RECEIPT_CONTENT_SHA256:
        raise Plan0071D2CohortError("D1 receipt authority drifted.")
    d0_manifest, exposure = _exposure_authority()
    exposed_documents = set(str(value) for value in exposure["document_ids"])
    exposed_hashes = set(str(value) for value in exposure["recording_hashes"])
    exposed_hashes.update(
        str(item["source_media_sha256"]) for item in exposure["full_recordings"]
    )
    units, total_count = _candidate_units(transcript_root)
    selection = select_structural_units(
        units,
        exposed_document_ids=exposed_documents,
        exposed_recording_hashes=exposed_hashes,
    )
    selected = selection["selected"]
    selected_documents = [str(item["document_id"]) for item in selected]
    selected_hashes = [str(item["source_media_sha256"]) for item in selected]
    if len(set(selected_documents)) != 6 or len(set(selected_hashes)) != 6:
        raise Plan0071D2CohortError("Supplemental cohort is not source-disjoint.")
    exclusion_documents = sorted(exposed_documents | set(selected_documents))
    exclusion_hashes = sorted(exposed_hashes | set(selected_hashes))
    database = transcript_root.expanduser().resolve() / "transcripts.sqlite3"
    snapshot = plan0064_p0._database_snapshot(
        database, ("documents", "document_blobs", "blobs")
    )
    return _content(
        {
            "schema_version": SCHEMA_VERSION,
            "status": "d2_supplemental_development_cohort_frozen_zero_effect",
            "source_authority": _source_authority(require_clean=True),
            "d0_activation_content_sha256": D0_ACTIVATION_CONTENT_SHA256,
            "d0_exposure_set_content_sha256": exposure["content_sha256"],
            "d1_receipt_content_sha256": D1_RECEIPT_CONTENT_SHA256,
            "transcript_database": str(database),
            "transcript_database_snapshot": snapshot,
            "selection_policy": {
                "order": "oldest_recording_time_then_document_id",
                "maximum_conversations": MAX_CONVERSATIONS,
                "required_speaker_count": REQUIRED_SPEAKER_COUNT,
                "requires_hash_matched_transcript": True,
                "requires_hash_matched_source_media": True,
                "requires_original_recording_filename": True,
                "excludes_prior_development_documents": True,
                "excludes_prior_development_recording_hashes": True,
                "uses_identity_gold": False,
                "uses_model_predictions": False,
                "uses_likely_pass_status": False,
            },
            "total_transcript_document_count": total_count,
            "considered_count": selection["considered_count"],
            "selected_count": selection["selected_count"],
            "last_considered_chronological_rank": selection[
                "last_considered_chronological_rank"
            ],
            "disposition_counts": selection["disposition_counts"],
            "reason_code_counts": selection["reason_code_counts"],
            "considered": selection["considered"],
            "selected_document_ids": selected_documents,
            "selected_recording_hashes": selected_hashes,
            "selected_original_recording_filenames": [
                str(item["original_recording_filename"]) for item in selected
            ],
            "original_recording_filename_count": 6,
            "development_exclusion_document_set_sha256": _hash(
                exclusion_documents
            ),
            "development_exclusion_recording_hash_set_sha256": _hash(
                exclusion_hashes
            ),
            "prediction_count": 0,
            "human_gold_read": False,
            "fresh_evaluation_allowed": False,
            "effect_counts": dict(EFFECT_COUNTS),
        }
    )


def _paths(runtime_root: Path, content_sha256: str) -> dict[str, Path]:
    root = runtime_root.expanduser().resolve()
    run = root / f"d2-cohort-{content_sha256[:24]}"
    return {
        "root": root,
        "run": run,
        "manifest": run / "private-manifest.json",
        "receipt": run / "receipt.json",
    }


def freeze_cohort(
    *,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    transcript_root: Path = DEFAULT_TRANSCRIPT_ROOT,
) -> dict[str, Any]:
    root = runtime_root.expanduser().resolve()
    existing = list(root.glob("d2-cohort-*/receipt.json"))
    if existing:
        return replay_cohort(runtime_root=root)
    manifest = build_cohort_manifest(transcript_root=transcript_root)
    paths = _paths(root, manifest["content_sha256"])
    if paths["run"].exists():
        raise Plan0071D2CohortError("A partial D2 cohort directory exists.")
    ensure_private_tree(paths["root"], paths["run"])
    write_immutable_private_json(paths["manifest"], manifest)
    receipt = _content(
        {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "status": "d2_cohort_frozen_zero_effect",
            "manifest_content_sha256": manifest["content_sha256"],
            "manifest_file_sha256": sha256_file(paths["manifest"]),
            "selected_count": 6,
            "speaker_slot_count": 18,
            "original_recording_filename_count": 6,
            "prediction_count": 0,
            "human_gold_read": False,
            "fresh_evaluation_allowed": False,
            "effect_counts": dict(EFFECT_COUNTS),
        }
    )
    write_immutable_private_json(paths["receipt"], receipt)
    return {
        **receipt,
        "manifest_path": str(paths["manifest"]),
        "receipt_path": str(paths["receipt"]),
        "idempotent_replay": False,
    }


def replay_cohort(*, runtime_root: Path = DEFAULT_RUNTIME_ROOT) -> dict[str, Any]:
    root = runtime_root.expanduser().resolve()
    receipts = list(root.glob("d2-cohort-*/receipt.json"))
    if len(receipts) != 1:
        raise Plan0071D2CohortError("Expected one Plan 0071 D2 cohort receipt.")
    receipt_path = receipts[0]
    manifest_path = receipt_path.with_name("private-manifest.json")
    receipt = read_private_object(receipt_path)
    manifest = read_private_object(manifest_path)
    _validate_content(receipt, "Plan 0071 D2 cohort receipt")
    _validate_content(manifest, "Plan 0071 D2 cohort manifest")
    for item in manifest.get("considered") or []:
        if item.get("disposition") != "selected_supplemental_development":
            continue
        for key in ("transcript_artifact", "source_media_artifact"):
            binding = item[key]
            if sha256_file(Path(str(binding["path"]))) != binding["file_sha256"]:
                raise Plan0071D2CohortError("A selected D2 artifact drifted.")
    current_source = _source_authority(require_clean=False)
    if (
        receipt.get("manifest_content_sha256") != manifest["content_sha256"]
        or receipt.get("manifest_file_sha256") != sha256_file(manifest_path)
        or manifest.get("selected_count") != 6
        or manifest.get("original_recording_filename_count") != 6
        or manifest.get("human_gold_read") is not False
        or manifest.get("fresh_evaluation_allowed") is not False
        or manifest.get("effect_counts") != EFFECT_COUNTS
        or manifest.get("source_authority", {}).get("module_sha256")
        != current_source.get("module_sha256")
    ):
        raise Plan0071D2CohortError("Plan 0071 D2 cohort replay drifted.")
    return {
        **receipt,
        "manifest_path": str(manifest_path),
        "receipt_path": str(receipt_path),
        "idempotent_replay": True,
    }


if __name__ == "__main__":
    print(json.dumps(freeze_cohort(), indent=2, sort_keys=True))
