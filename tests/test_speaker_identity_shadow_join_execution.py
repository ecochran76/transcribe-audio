from __future__ import annotations

import json
from pathlib import Path

import pytest

from acoustic_audio_derivatives import (
    canonical_artifact_hash,
    ensure_private_tree,
    sha256_file,
    write_immutable_private_json,
)
from speaker_identity_orchestration import IdentityOrchestrationError, negative_action_vector
from speaker_identity_shadow_join_execution import (
    ACTIVATION_VERSION,
    _lane_paths,
    replay_plan0060_activation,
)


def _activation(root: Path) -> tuple[str, Path]:
    manifest = {
        "schema_version": ACTIVATION_VERSION,
        "status": "activated_pre_implementation",
        "activated_at": "2026-08-08T12:00:00-05:00",
        "negative_actions": negative_action_vector(),
    }
    content_sha256 = canonical_artifact_hash(manifest)
    run = root / f"activation-{content_sha256[:24]}"
    ensure_private_tree(root, run)
    manifest_path = run / "private-manifest.json"
    write_immutable_private_json(manifest_path, manifest)
    write_immutable_private_json(
        run / "receipt.json",
        {
            "content_sha256": content_sha256,
            "manifest_sha256": sha256_file(manifest_path),
            "recording_count": 3,
            "speaker_ref_count": 10,
            "inherited_replay_verified": True,
            "negative_actions_preserved": True,
        },
    )
    return content_sha256, manifest_path


def test_independent_lane_paths_cannot_collide(tmp_path: Path) -> None:
    acoustic = _lane_paths(tmp_path, "a" * 64, "p2a-acoustic")
    context = _lane_paths(tmp_path, "a" * 64, "p2b-context")

    assert acoustic["run"] != context["run"]
    assert acoustic["manifest"] != context["manifest"]


def test_activation_replay_binds_complete_denominator(tmp_path: Path) -> None:
    content_sha256, _ = _activation(tmp_path)

    replay = replay_plan0060_activation(
        runtime_root=tmp_path,
        activation_sha256=content_sha256,
    )

    assert replay["receipt"]["recording_count"] == 3
    assert replay["receipt"]["speaker_ref_count"] == 10
    assert replay["idempotent_replay"] is True


def test_activation_replay_rejects_manifest_tamper(tmp_path: Path) -> None:
    content_sha256, manifest_path = _activation(tmp_path)
    manifest_path.chmod(0o600)
    manifest_path.write_text(json.dumps({"status": "tampered"}), encoding="utf-8")

    with pytest.raises(IdentityOrchestrationError) as excinfo:
        replay_plan0060_activation(
            runtime_root=tmp_path,
            activation_sha256=content_sha256,
        )

    assert excinfo.value.reason_code == "plan0060_activation_invalid"


def test_unknown_lane_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(IdentityOrchestrationError) as excinfo:
        _lane_paths(tmp_path, "a" * 64, "p2")

    assert excinfo.value.reason_code == "invalid_plan0060_lane"
