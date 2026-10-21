from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from acoustic_verification import FakeVerificationAdapter
from speaker_match_bridge import ComparisonBridge
from test_speaker_enrollment_bridge import fixture, request


def prepared(tmp_path):
    enrollment, config = fixture(tmp_path)
    job = enrollment.request(request())
    enrollment.close()
    bridge = ComparisonBridge(
        config, FakeVerificationAdapter(candidate_id="synthetic_adapter")
    )
    profile = {
        key: job[key]
        for key in (
            "person",
            "profile_version_id",
            "profile_sha256",
            "adapter_id",
            "model_revision",
        )
    }
    probe = request("probe")
    return bridge, {
        "action": "compare",
        "tenant": "one",
        "app": "speech",
        "job_id": "comparison",
        "samples": probe["samples"],
        "profiles": [profile],
    }


def test_comparison_is_bounded_uncalibrated_read_only_evidence(tmp_path):
    bridge, payload = prepared(tmp_path)
    try:
        before = bridge.request(
            {"action": "profiles", "tenant": "one", "app": "speech"}
        )
        result = bridge.request(payload)
        assert result["id"] == "comparison"
        assert result["identity_assigned"] is False
        assert result["enrollment_performed"] is False
        assert result["qualified"] is False
        assert result["candidates"][0]["score"] == pytest.approx(1.0)
        assert result["candidates"][0]["score_kind"] == "uncalibrated_cosine"
        assert (
            result["candidates"][0]["profile_sha256"]
            == payload["profiles"][0]["profile_sha256"]
        )
        encoded = json.dumps(result)
        assert (
            "data_base64" not in encoded
            and '"centroid"' not in encoded
            and str(tmp_path) not in encoded
        )
        assert (
            bridge.request({"action": "profiles", "tenant": "one", "app": "speech"})
            == before
        )
    finally:
        bridge.close()


@pytest.mark.parametrize(
    "change,code",
    [
        (lambda r: r.update(app="foreign"), "forbidden"),
        (
            lambda r: r["profiles"][0]["person"].update(revision="changed"),
            "authority_drift",
        ),
        (lambda r: r["profiles"][0].update(profile_sha256="a" * 64), "profile_drift"),
        (lambda r: r["profiles"][0].update(model_revision="changed"), "model_drift"),
        (lambda r: r["samples"][0].update(sha256="a" * 64), "sample_hash_mismatch"),
        (lambda r: r["samples"].append(dict(r["samples"][0])), "one_probe_required"),
    ],
)
def test_unscoped_or_changed_evidence_is_rejected_before_embedding(
    tmp_path, change, code
):
    bridge, payload = prepared(tmp_path)

    def forbidden_embed(*args, **kwargs):
        pytest.fail("Rejected evidence must never reach embedding")

    bridge.adapter.embed = forbidden_embed
    try:
        change(payload)
        with pytest.raises(ValueError, match=code):
            bridge.request(payload)
    finally:
        bridge.close()


def test_replaced_head_is_rejected_even_when_old_version_remains_active(tmp_path):
    bridge, payload = prepared(tmp_path)
    try:
        old = payload["profiles"][0]["profile_version_id"]
        replacement = bridge.enrollment.request(request("replacement", old))
        assert replacement["state"] == "completed"
        assert bridge.enrollment.custody.profile_state(old)["status"] == "active"
        with pytest.raises(ValueError, match="profile_not_current"):
            bridge.request(payload)
    finally:
        bridge.close()


def test_profile_invalidation_during_embedding_discards_all_scores(tmp_path):
    bridge, payload = prepared(tmp_path)
    original = bridge.adapter.embed

    def invalidate(*args, **kwargs):
        bridge.enrollment.custody.record_profile_event(
            profile_version_id=payload["profiles"][0]["profile_version_id"],
            supersedes_event_id=bridge.enrollment.custody.profile_state(
                payload["profiles"][0]["profile_version_id"]
            )["event_id"],
            action="invalidate",
            reason_code="synthetic_test_withdrawal",
            authority_id="enroll-1",
            idempotency_key="invalidate",
            created_at="2026-10-21T00:00:00Z",
        )
        return original(*args, **kwargs)

    bridge.adapter.embed = invalidate
    try:
        with pytest.raises(ValueError, match="profile_not_active"):
            bridge.request(payload)
    finally:
        bridge.close()


def test_changed_private_profile_bytes_are_not_used(tmp_path):
    bridge, payload = prepared(tmp_path)
    try:
        obj = (
            bridge.enrollment.private_root
            / "objects"
            / ("profile-" + payload["profiles"][0]["profile_sha256"])
        )
        obj.write_bytes(b"corrupted")
        with pytest.raises(ValueError, match="profile_hash_mismatch"):
            bridge.request(payload)
    finally:
        bridge.close()


def test_changed_review_authority_and_excluded_samples_cannot_be_scored(tmp_path):
    bridge, payload = prepared(tmp_path)
    try:
        bridge.enrollment.config["authorities"]["person"]["review_id"] = (
            "different-review"
        )
        with pytest.raises(ValueError, match="authority_drift"):
            bridge.request(payload)
        bridge.enrollment.config["authorities"]["person"]["review_id"] = "review-1"
        sample_id = bridge.enrollment.custody.profile_state(
            payload["profiles"][0]["profile_version_id"]
        )["sample_ids"][0]
        bridge.enrollment.custody.record_sample_event(
            sample_id=sample_id,
            event_type="exclude",
            actor_id="tester",
            authority_id="enroll-1",
            idempotency_key="exclude",
            created_at="2026-10-21T00:00:00Z",
        )
        with pytest.raises(ValueError, match="profile_sample_unavailable"):
            bridge.request(payload)
    finally:
        bridge.close()


def test_production_cli_reports_missing_assets_without_loading_or_accepting_fake_adapter(
    tmp_path,
):
    import os
    import subprocess
    from acoustic_verification import EXPECTED_CANDIDATES

    bridge, _ = prepared(tmp_path)
    config = dict(bridge.config)
    bridge.close()
    filename = tmp_path / "private-config.json"
    filename.write_text(json.dumps(config))
    filename.chmod(0o600)
    command = [
        sys.executable,
        "speaker_match_bridge.py",
        "--config",
        str(filename),
        "--parent-pid",
        str(os.getpid()),
    ]
    read = {"action": "readiness", "tenant": "one", "app": "speech"}
    rejected = subprocess.run(
        command, input=json.dumps(read), capture_output=True, text=True, timeout=10
    )
    assert rejected.returncode == 1
    assert json.loads(rejected.stdout)["ok"] is False
    config["adapter_id"] = "speechbrain_ecapa_tdnn"
    config["model_revision"] = EXPECTED_CANDIDATES["speechbrain_ecapa_tdnn"][
        "revision_sha"
    ]
    filename.write_text(json.dumps(config))
    ready = subprocess.run(
        command, input=json.dumps(read), capture_output=True, text=True, timeout=10
    )
    assert ready.returncode == 0, ready.stdout
    result = json.loads(ready.stdout)["result"]
    assert result["available"] is False and result["qualified"] is False
    assert str(tmp_path) not in ready.stdout
    denied = subprocess.run(
        command,
        input=json.dumps({**read, "tenant": "other"}),
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert json.loads(denied.stdout) == {"ok": False, "error": "comparison_forbidden"}
