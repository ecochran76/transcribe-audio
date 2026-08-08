from __future__ import annotations

from pathlib import Path

from acoustic_audio_derivatives import (
    canonical_artifact_hash,
    ensure_private_tree,
    sha256_file,
    write_immutable_private_json,
)
from speaker_identity_orchestration import negative_action_vector
from speaker_identity_shadow_review import (
    ACTIVATION_SHA256,
    MANIFEST_VERSION,
    load_joined_shadow_review,
)


def _packet(state_root: Path) -> str:
    root = state_root / "plan-0060"
    run = root / f"p4-review-packet-{ACTIVATION_SHA256[:24]}"
    ensure_private_tree(root, run)
    manifest = {
        "schema_version": MANIFEST_VERSION,
        "status": "sealed_pending_human_review",
        "activation_sha256": ACTIVATION_SHA256,
        "join_content_sha256": "a" * 64,
        "human_decision_count": 0,
        "preselected_decision_count": 0,
        "apply_enabled": False,
        "human_gold_read": False,
        "cases": [
            {
                "document_id": "document-test",
                "candidate_options": [{"person_id": "person-test", "label": "Candidate"}],
                "scopes": [],
                "warnings": [],
                "source_failures": [],
                "speaker_slots": [
                    {
                        "speaker_ref": "SPEAKER_1",
                        "selected_person_id": None,
                        "conditions": [{"condition": "combined"}],
                    }
                ],
            }
        ],
        "negative_actions": negative_action_vector(),
    }
    content_sha256 = canonical_artifact_hash(manifest)
    manifest_path = run / "private-manifest.json"
    write_immutable_private_json(manifest_path, manifest)
    write_immutable_private_json(
        run / "receipt.json",
        {
            "content_sha256": content_sha256,
            "manifest_sha256": sha256_file(manifest_path),
        },
    )
    return content_sha256


def test_joined_shadow_loader_exposes_sealed_case_without_apply(tmp_path: Path) -> None:
    expected_hash = _packet(tmp_path)

    review = load_joined_shadow_review(
        document_id="document-test",
        state_root=tmp_path,
    )

    assert review["status"] == "sealed_pending_human_review"
    assert review["packet_content_sha256"] == expected_hash
    assert review["speaker_slots"][0]["selected_person_id"] is None
    assert review["apply_enabled"] is False
    assert review["will_perform_external_write"] is False


def test_joined_shadow_loader_does_not_leak_other_cohort_case(tmp_path: Path) -> None:
    _packet(tmp_path)

    review = load_joined_shadow_review(
        document_id="different-document",
        state_root=tmp_path,
    )

    assert review["status"] == "not_in_frozen_cohort"
    assert "speaker_slots" not in review
    assert review["apply_enabled"] is False
