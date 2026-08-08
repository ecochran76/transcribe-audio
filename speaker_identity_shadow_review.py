from __future__ import annotations

from pathlib import Path
from typing import Any

from acoustic_audio_derivatives import (
    canonical_artifact_hash,
    read_private_object,
    require_private_file,
    sha256_file,
)


ACTIVATION_SHA256 = "08afc1b021a30f2a06f6e45bac88cec1b343def65b4e02261845ddff8667cf77"
MANIFEST_VERSION = "transcribe-audio.plan0060-p4-review-packet-manifest.v1"
RUNTIME_DIRNAME = "plan-0060"


def _paths(state_root: Path) -> tuple[Path, Path, Path]:
    root = state_root.expanduser().absolute() / RUNTIME_DIRNAME
    run = root / f"p4-review-packet-{ACTIVATION_SHA256[:24]}"
    return root, run / "private-manifest.json", run / "receipt.json"


def load_joined_shadow_review(
    *, document_id: str, state_root: Path
) -> dict[str, Any]:
    """Read one sealed Plan 0060 case without exposing an apply path."""

    root, manifest_path, receipt_path = _paths(state_root)
    if not manifest_path.exists() or not receipt_path.exists():
        return {
            "status": "absent",
            "source_document_id": document_id,
            "apply_enabled": False,
            "will_perform_external_write": False,
        }
    require_private_file(manifest_path, root)
    require_private_file(receipt_path, root)
    manifest = read_private_object(manifest_path)
    receipt = read_private_object(receipt_path)
    if (
        manifest.get("schema_version") != MANIFEST_VERSION
        or manifest.get("activation_sha256") != ACTIVATION_SHA256
        or receipt.get("content_sha256") != canonical_artifact_hash(manifest)
        or receipt.get("manifest_sha256") != sha256_file(manifest_path)
        or manifest.get("status") != "sealed_pending_human_review"
        or manifest.get("apply_enabled") is not False
        or manifest.get("human_decision_count") != 0
        or manifest.get("preselected_decision_count") != 0
        or manifest.get("human_gold_read") is not False
        or any((manifest.get("negative_actions") or {}).values())
    ):
        return {
            "status": "rejected",
            "reason": "sealed_packet_binding_invalid",
            "source_document_id": document_id,
            "apply_enabled": False,
            "will_perform_external_write": False,
        }
    case = next(
        (
            item
            for item in manifest.get("cases") or []
            if str(item.get("document_id") or "") == document_id
        ),
        None,
    )
    if case is None:
        return {
            "status": "not_in_frozen_cohort",
            "source_document_id": document_id,
            "packet_content_sha256": receipt["content_sha256"],
            "apply_enabled": False,
            "will_perform_external_write": False,
        }
    return {
        "schema_version": MANIFEST_VERSION,
        "status": "sealed_pending_human_review",
        "source_document_id": document_id,
        "packet_content_sha256": receipt["content_sha256"],
        "join_content_sha256": manifest["join_content_sha256"],
        "candidate_options": case["candidate_options"],
        "scopes": case["scopes"],
        "warnings": case["warnings"],
        "source_failures": case["source_failures"],
        "speaker_slots": case["speaker_slots"],
        "human_decision_count": 0,
        "preselected_decision_count": 0,
        "apply_enabled": False,
        "will_perform_external_write": False,
    }
