from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import wave

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from acoustic_verification import FakeVerificationAdapter
from speaker_enrollment_bridge import EnrollmentBridge, EnrollmentError


def wav():
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(b"\x00\x01" * 16_000)
    return output.getvalue()


def fixture(tmp_path):
    adapter = FakeVerificationAdapter(candidate_id="synthetic_adapter")
    config = {
        "directory": str(tmp_path / "bridge"),
        "tenant": "one",
        "apps": ["speech"],
        "adapter_id": adapter.candidate_id,
        "model_revision": adapter.revision_sha,
        "model_snapshot_root": str(tmp_path / "unconfigured-models"),
        "authorities": {
            "person": {
                "person_revision": "r1",
                "review_id": "review-1",
                "enrollment_id": "enroll-1",
            }
        },
    }
    return EnrollmentBridge(config, adapter), config


def request(job_id="job", predecessor=None):
    data = wav()
    return {
        "action": "enroll",
        "tenant": "one",
        "app": "speech",
        "job_id": job_id,
        "person": {"id": "person", "revision": "r1"},
        "authority": {
            "review_id": "review-1",
            "enrollment_id": "enroll-1",
            "reviewed_identity": True,
            "retained": True,
        },
        "expected_predecessor": predecessor,
        "samples": [
            {
                "id": "sample",
                "revision": 1,
                "sha256": hashlib.sha256(data).hexdigest(),
                "data_base64": base64.b64encode(data).decode(),
                "source": {
                    "recording_id": "recording",
                    "sha256": "a" * 64,
                    "start_ms": 0,
                    "end_ms": 1000,
                },
            }
        ],
    }


def test_enroll_rebuild_retains_checked_immutable_versions_and_private_custody(
    tmp_path,
):
    bridge, config = fixture(tmp_path)
    try:
        result = bridge.request(request())
        assert result["state"] == "completed", result
        assert result["qualified"] is False
        assert '"centroid":' not in json.dumps(result)
        assert "data_base64" not in json.dumps(result)
        assert str(tmp_path) not in json.dumps(result)
        assert bridge.request(request()) == result
        second = bridge.request(request("rebuild", result["profile_version_id"]))
        assert second["state"] == "completed", second
        assert second["predecessor_profile_version_id"] == result["profile_version_id"]
        assert (
            bridge.custody.profile_state(result["profile_version_id"])["status"]
            == "active"
        )
        assert (
            bridge.request({"action": "profiles", "tenant": "one", "app": "speech"})[
                "profiles"
            ][0]["profile_version_id"]
            == second["profile_version_id"]
        )
    finally:
        bridge.close()
    reopened = EnrollmentBridge(
        config, FakeVerificationAdapter(candidate_id="synthetic_adapter")
    )
    assert (
        reopened.request(
            {"action": "status", "tenant": "one", "app": "speech", "job_id": "job"}
        )
        == result
    )
    reopened.close()


def test_authority_scope_hash_predecessor_failures_do_not_enroll(tmp_path):
    bridge, _ = fixture(tmp_path)
    try:
        for mutate, code in [
            (lambda r: r.update(tenant="other"), "forbidden"),
            (
                lambda r: r["authority"].update(reviewed_identity=False),
                "authority_required",
            ),
            (
                lambda r: r["authority"].update(enrollment_id="invented"),
                "authority_drift",
            ),
            (lambda r: r["samples"][0].update(sha256="b" * 64), "sample_hash_mismatch"),
            (lambda r: r.update(expected_predecessor="absent"), "predecessor_conflict"),
        ]:
            item = request()
            mutate(item)
            with pytest.raises(EnrollmentError, match=code):
                bridge.request(item)
        assert bridge.db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0
    finally:
        bridge.close()


def test_failed_rebuild_preserves_prior_head_and_interruption_never_replays(tmp_path):
    bridge, config = fixture(tmp_path)
    result = bridge.request(request())
    assert result["state"] == "completed", result

    def failed(*args, **kwargs):
        raise RuntimeError("private model details")

    bridge.adapter.embed = failed
    failed_result = bridge.request(request("failed", result["profile_version_id"]))
    assert failed_result["state"] == "failed"
    assert failed_result["error"] == "enrollment_build_failed"
    assert (
        bridge.request({"action": "profiles", "tenant": "one", "app": "speech"})[
            "profiles"
        ][0]["id"]
        == "job"
    )
    bridge.db.execute(
        "UPDATE jobs SET payload=? WHERE id='failed'",
        (json.dumps({**failed_result, "state": "running"}),),
    )
    bridge.db.commit()
    bridge.close()
    reopened = EnrollmentBridge(
        config, FakeVerificationAdapter(candidate_id="synthetic_adapter")
    )
    status = reopened.request(
        {"action": "status", "tenant": "one", "app": "speech", "job_id": "failed"}
    )
    assert status["state"] == "unknown"
    assert (
        reopened.request(request("failed", result["profile_version_id"]))["state"]
        == "unknown"
    )
    with pytest.raises(EnrollmentError, match="busy_or_unknown"):
        reopened.request(request("next", result["profile_version_id"]))
    reopened.close()


def test_cancel_from_another_connection_waits_for_quiescence_preserves_head(tmp_path):
    bridge, config = fixture(tmp_path)
    other = EnrollmentBridge(
        config, FakeVerificationAdapter(candidate_id="synthetic_adapter")
    )
    original = bridge.adapter.embed

    def cancelled(*args, **kwargs):
        response = other.request(
            {"action": "cancel", "tenant": "one", "app": "speech", "job_id": "job"}
        )
        assert response["state"] == "running"
        assert response["cancellation"] == "waiting_for_quiescence"
        return original(*args, **kwargs)

    bridge.adapter.embed = cancelled
    result = bridge.request(request())
    assert result["state"] == "cancelled", result
    assert bridge.request({"action": "profiles", "tenant": "one", "app": "speech"}) == {
        "profiles": []
    }
    bridge.close()
    other.close()


@pytest.mark.parametrize("interrupted_state", ["running", "unknown"])
def test_lost_promotion_acknowledgement_recovers_exact_custody_event_without_embedding(
    tmp_path,
    interrupted_state,
):
    bridge, config = fixture(tmp_path)
    result = bridge.request(request())
    assert result["state"] == "completed"
    interrupted = {**result, "state": interrupted_state}
    interrupted.pop("profile_version_id")
    interrupted.pop("profile_sha256")
    bridge.db.execute(
        "UPDATE jobs SET payload=? WHERE id='job'", (json.dumps(interrupted),)
    )
    bridge.db.execute("DELETE FROM heads")
    bridge.db.commit()
    bridge.close()
    reopened = EnrollmentBridge(
        config, FakeVerificationAdapter(candidate_id="synthetic_adapter")
    )

    def unexpected(*args, **kwargs):
        pytest.fail("Reconciliation must never rerun embeddings")

    reopened.adapter.embed = unexpected
    recovered = reopened.request(
        {"action": "status", "tenant": "one", "app": "speech", "job_id": "job"}
    )
    assert recovered["state"] == "completed"
    assert recovered["profile_version_id"] == result["profile_version_id"]
    assert (
        reopened.request({"action": "profiles", "tenant": "one", "app": "speech"})[
            "profiles"
        ][0]["profile_version_id"]
        == result["profile_version_id"]
    )
    reopened.close()


def test_cli_parent_guard_refuses_orphaned_invocation_before_storage(tmp_path):
    config = tmp_path / "private-config.json"
    config.write_text("{}")
    config.chmod(0o600)
    command = [
        sys.executable,
        "speaker_enrollment_bridge.py",
        "--config",
        str(config),
        "--parent-pid",
        "2147483647",
    ]
    result = subprocess.run(
        command, input="{}", text=True, capture_output=True, timeout=10
    )
    assert result.returncode == 1
    assert json.loads(result.stdout) == {
        "ok": False,
        "error": "enrollment_parent_guard_unavailable",
    }
    assert list(tmp_path.iterdir()) == [config]


def test_process_readiness_is_supported_without_loading_models_and_denies_unscoped_effects(
    tmp_path,
):
    from acoustic_verification import EXPECTED_CANDIDATES

    config = tmp_path / "private-config.json"
    root = tmp_path / "new-custody"
    value = {
        "directory": str(root),
        "tenant": "one",
        "apps": ["speech"],
        "adapter_id": "speechbrain_ecapa_tdnn",
        "model_revision": EXPECTED_CANDIDATES["speechbrain_ecapa_tdnn"]["revision_sha"],
        "model_snapshot_root": str(tmp_path / "not-installed" / "models"),
        "authorities": {},
    }
    config.write_text(json.dumps(value))
    config.chmod(0o600)
    command = [sys.executable, "speaker_enrollment_bridge.py", "--config", str(config)]
    denied = subprocess.run(
        command,
        input=json.dumps({"action": "readiness", "tenant": "other", "app": "speech"}),
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert json.loads(denied.stdout) == {"ok": False, "error": "enrollment_forbidden"}
    assert not root.exists()
    read = subprocess.run(
        command,
        input=json.dumps({"action": "readiness", "tenant": "one", "app": "speech"}),
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert read.returncode == 0, read.stdout
    result = json.loads(read.stdout)
    assert result["ok"] is True
    assert result["result"]["available"] is False
    assert result["result"]["qualified"] is False
    assert str(tmp_path) not in read.stdout


def test_two_distinct_references_preserve_both_hashes_and_build_one_profile(tmp_path):
    bridge, _ = fixture(tmp_path)
    item = request()
    second_bytes = bytearray(wav())
    second_bytes[44:] = b"\x00\x02" * 16_000
    second = {
        **item["samples"][0],
        "id": "second-sample",
        "sha256": hashlib.sha256(second_bytes).hexdigest(),
        "data_base64": base64.b64encode(second_bytes).decode(),
        "source": {**item["samples"][0]["source"], "start_ms": 2000, "end_ms": 3000},
    }
    item["samples"].append(second)
    try:
        result = bridge.request(item)
        assert result["state"] == "completed", result
        assert {ref["sha256"] for ref in result["sample_manifest"]} == {
            item["samples"][0]["sha256"],
            second["sha256"],
        }
        assert len({ref["sha256"] for ref in result["sample_manifest"]}) == 2
        state = bridge.custody.profile_state(result["profile_version_id"])
        assert len(state["sample_ids"]) == 2
        assert state["status"] == "active"
        selected = bridge.request(
            {"action": "profiles", "tenant": "one", "app": "speech"}
        )
        assert (
            selected["profiles"][0]["profile_version_id"]
            == result["profile_version_id"]
        )
    finally:
        bridge.close()
