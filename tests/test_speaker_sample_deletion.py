from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from acoustic_verification import FakeVerificationAdapter
from speaker_enrollment_bridge import EnrollmentBridge, EnrollmentError
from speaker_sample_deletion import SampleDeletionBridge, SampleSetCustodyLedger
from tests.test_speaker_enrollment_bridge import fixture, request, wav
from speaker_match_bridge import ComparisonBridge


def setup(tmp_path):
    bridge, config = fixture(tmp_path)
    bridge.config["deletion_authority_id"] = "explicit-operator-delete"
    config["deletion_authority_id"] = "explicit-operator-delete"
    item = request()
    other = bytearray(wav())
    other[44:] = b"\x00\x02" * 16_000
    item["samples"].append(
        {
            **item["samples"][0],
            "id": "second",
            "sha256": hashlib.sha256(other).hexdigest(),
            "data_base64": base64.b64encode(other).decode(),
            "source": {
                **item["samples"][0]["source"],
                "start_ms": 2000,
                "end_ms": 3000,
            },
        }
    )
    initial = bridge.request(item)
    assert initial["state"] == "completed"
    rebuilt = {
        **item,
        "job_id": "rebuild",
        "expected_predecessor": initial["profile_version_id"],
    }
    second = bridge.request(rebuilt)
    assert second["state"] == "completed"
    return bridge, config, item, initial, second


def wire(action, **values):
    return {"tenant": "one", "app": "speech", "action": action, **values}


def test_aggregate_deletes_all_shared_profile_versions_and_private_bytes_with_exact_replay(
    tmp_path,
):
    bridge, _, item, first, second = setup(tmp_path)
    preview = bridge.request(wire("deletion_preview", sample_ids=["sample", "second"]))
    assert preview["available"] is True
    assert {p["profile_version_id"] for p in preview["profiles"]} == {
        first["profile_version_id"],
        second["profile_version_id"],
    }
    assert preview["operations"] == []
    operation = wire(
        "deletion_apply",
        operation_id="delete",
        sample_ids=["sample", "second"],
        fingerprint=preview["fingerprint"],
        approve=True,
    )
    result = bridge.request(operation)
    assert result["status"] == "applied", result
    assert result["private_cleanup_verified"] is True
    assert bridge.request(operation) == result
    assert bridge.request(wire("deletion_status", operation_id="delete")) == result
    assert bridge.request(wire("profiles")) == {"profiles": []}
    assert list((bridge.private_root / "objects").iterdir()) == []
    assert all(
        bridge.custody.profile_state(p)["status"] == "invalidated"
        for p in [first["profile_version_id"], second["profile_version_id"]]
    )
    with pytest.raises(EnrollmentError, match="reference_deleted"):
        bridge.request(
            {**item, "job_id": "must-not-restore", "expected_predecessor": None}
        )
    with pytest.raises(EnrollmentError, match="request_conflict"):
        bridge.request({**operation, "sample_ids": ["sample"]})
    bridge.close()


def test_scoped_authority_stale_preview_and_shared_profile_sequential_deletion(
    tmp_path,
):
    bridge, _, _, _, _ = setup(tmp_path)
    preview = bridge.request(wire("deletion_preview", sample_ids=["sample"]))
    with pytest.raises(EnrollmentError, match="forbidden"):
        bridge.request(
            {
                **wire(
                    "deletion_apply",
                    operation_id="delete",
                    sample_ids=["sample"],
                    fingerprint=preview["fingerprint"],
                    approve=True,
                ),
                "app": "other",
            }
        )
    with pytest.raises(EnrollmentError, match="preview_stale"):
        bridge.request(
            wire(
                "deletion_apply",
                operation_id="delete",
                sample_ids=["sample"],
                fingerprint="0" * 64,
                approve=True,
            )
        )
    del bridge.config["deletion_authority_id"]
    assert (
        bridge.request(wire("deletion_preview", sample_ids=["sample"]))["available"]
        is False
    )
    with pytest.raises(EnrollmentError, match="authority_unconfigured"):
        bridge.request(
            wire(
                "deletion_apply",
                operation_id="delete",
                sample_ids=["sample"],
                fingerprint=preview["fingerprint"],
                approve=True,
            )
        )
    bridge.config["deletion_authority_id"] = "explicit-operator-delete"
    first = bridge.request(
        wire(
            "deletion_apply",
            operation_id="delete",
            sample_ids=["sample"],
            fingerprint=preview["fingerprint"],
            approve=True,
        )
    )
    assert first["status"] == "applied"
    remaining = bridge.request(wire("deletion_preview", sample_ids=["second"]))
    last = bridge.request(
        wire(
            "deletion_apply",
            operation_id="delete-second",
            sample_ids=["second"],
            fingerprint=remaining["fingerprint"],
            approve=True,
        )
    )
    assert last["status"] == "applied", last
    assert list((bridge.private_root / "objects").iterdir()) == []
    bridge.close()


def test_lost_native_ack_reconciles_tombstone_and_cleanup_without_reapplying(
    tmp_path, monkeypatch
):
    bridge, _, _, _, _ = setup(tmp_path)
    preview = bridge.request(wire("deletion_preview", sample_ids=["sample", "second"]))
    original = SampleSetCustodyLedger.apply_effect

    def lost(self, **kwargs):
        original(self, **kwargs)
        raise RuntimeError("lost terminal acknowledgement")

    monkeypatch.setattr(SampleSetCustodyLedger, "apply_effect", lost)
    operation = wire(
        "deletion_apply",
        operation_id="delete",
        sample_ids=["sample", "second"],
        fingerprint=preview["fingerprint"],
        approve=True,
    )
    assert bridge.request(operation)["status"] == "unknown"

    def forbidden(*args, **kwargs):
        pytest.fail("Unknown effect must not be replayed")

    monkeypatch.setattr(SampleSetCustodyLedger, "apply_effect", forbidden)
    result = bridge.request(wire("deletion_status", operation_id="delete"))
    assert result["status"] == "applied", result
    assert bridge.request(operation) == result
    bridge.close()


def test_unconfirmed_effect_retains_fence_and_unknown_without_replay(
    tmp_path, monkeypatch
):
    bridge, config, item, _, _ = setup(tmp_path)
    preview = bridge.request(wire("deletion_preview", sample_ids=["sample"]))

    def interrupted(*args, **kwargs):
        raise RuntimeError("before effect receipt")

    monkeypatch.setattr(SampleSetCustodyLedger, "apply_effect", interrupted)
    operation = wire(
        "deletion_apply",
        operation_id="delete",
        sample_ids=["sample"],
        fingerprint=preview["fingerprint"],
        approve=True,
    )
    assert bridge.request(operation)["status"] == "unknown"
    assert (
        bridge.request(wire("deletion_status", operation_id="delete"))["status"]
        == "unknown"
    )
    assert bridge.request(operation)["status"] == "unknown"
    with pytest.raises(EnrollmentError, match="reference_deleted"):
        bridge.request({**item, "job_id": "new", "expected_predecessor": None})
    bridge.close()


def test_orphan_pre_registration_cleanup_reconciles_durable_approved_intent(
    tmp_path, monkeypatch
):
    bridge, _ = fixture(tmp_path)
    bridge.config["deletion_authority_id"] = "explicit-operator-delete"

    def interrupted(*args, **kwargs):
        raise RuntimeError("before sample registration")

    monkeypatch.setattr(bridge.custody, "register_sample", interrupted)
    assert bridge.request(request())["state"] == "failed"
    assert list((bridge.private_root / "objects").iterdir())
    preview = bridge.request(wire("deletion_preview", sample_ids=["sample"]))
    assert preview["samples"][0]["upstream_sample_ids"] == []
    original = SampleDeletionBridge._cleanup

    def lost(*args, **kwargs):
        raise RuntimeError("interrupt cleanup")

    monkeypatch.setattr(SampleDeletionBridge, "_cleanup", lost)
    operation = wire(
        "deletion_apply",
        operation_id="orphan-delete",
        sample_ids=["sample"],
        fingerprint=preview["fingerprint"],
        approve=True,
    )
    assert bridge.request(operation)["status"] == "unknown"
    monkeypatch.setattr(SampleDeletionBridge, "_cleanup", original)
    result = bridge.request(wire("deletion_status", operation_id="orphan-delete"))
    assert result["status"] == "applied", result
    assert result["tombstone_id"] is None
    assert result["private_cleanup_verified"] is True
    assert list((bridge.private_root / "objects").iterdir()) == []
    assert bridge.request(operation) == result
    bridge.close()
