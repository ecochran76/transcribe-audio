from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from acoustic_verification import (
    AcousticVerificationError,
    dry_run_model_acquisition,
    replay_model_acquisition,
)


def test_model_acquisition_dry_run_is_immutable_and_side_effect_free(
    tmp_path: Path,
) -> None:
    root = tmp_path / "p4-acquisition"

    plan = dry_run_model_acquisition(runtime_root=root)

    assert plan["status"] == "success"
    assert plan["reason_code"] is None
    assert plan["authorization_basis"] == "operator_blanket_2026-07-31"
    assert plan["spec"]["authorization_scope"] == (
        "plan_0037_model_acquisition_install_and_development_processing_only"
    )
    assert plan["spec"]["real_biometric_enrollment_authorized"] is False
    assert [item["candidate_id"] for item in plan["spec"]["candidates"]] == [
        "speechbrain_ecapa_tdnn",
        "wespeaker_campplus",
        "wespeaker_resnet34",
    ]
    for field in (
        "will_download",
        "will_install",
        "will_build",
        "will_read_audio",
        "will_materialize_embeddings",
        "will_register_references",
        "will_run_trials",
        "will_perform_external_write",
    ):
        assert plan[field] is False

    replay = replay_model_acquisition(
        plan["run_id"],
        expected_dry_run_sha256=plan["dry_run_sha256"],
        runtime_root=root,
    )
    assert replay["dry_run_sha256"] == plan["dry_run_sha256"]
    assert replay["spec_sha256"] == plan["spec_sha256"]
    for path in root.rglob("*"):
        expected = 0o700 if path.is_dir() else 0o600
        assert stat.S_IMODE(path.stat().st_mode) == expected


def test_model_acquisition_replay_rejects_spec_drift(tmp_path: Path) -> None:
    source = (
        Path(__file__).parents[1]
        / "docs/dev/fixtures/plan-0037-p4/verification-model-acquisition-plan.json"
    )
    spec = tmp_path / "acquisition.json"
    spec.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    root = tmp_path / "p4-acquisition"
    plan = dry_run_model_acquisition(runtime_root=root, spec_path=spec)

    payload = json.loads(spec.read_text(encoding="utf-8"))
    payload["candidates"][0]["model"]["revision_sha"] = "0" * 40
    spec.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AcousticVerificationError, match="spec drifted"):
        replay_model_acquisition(
            plan["run_id"],
            expected_dry_run_sha256=plan["dry_run_sha256"],
            runtime_root=root,
        )


def test_model_acquisition_rejects_real_enrollment_authority(
    tmp_path: Path,
) -> None:
    source = (
        Path(__file__).parents[1]
        / "docs/dev/fixtures/plan-0037-p4/verification-model-acquisition-plan.json"
    )
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["real_biometric_enrollment_authorized"] = True
    spec = tmp_path / "acquisition.json"
    spec.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AcousticVerificationError, match="enrollment authority"):
        dry_run_model_acquisition(
            runtime_root=tmp_path / "p4-acquisition", spec_path=spec
        )


def test_model_acquisition_rejects_mutable_terms_authority(tmp_path: Path) -> None:
    source = (
        Path(__file__).parents[1]
        / "docs/dev/fixtures/plan-0037-p4/verification-model-acquisition-plan.json"
    )
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["source_authorities"]["wespeaker_models"] = (
        "https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md"
    )
    spec = tmp_path / "acquisition.json"
    spec.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AcousticVerificationError, match="authority binding"):
        dry_run_model_acquisition(
            runtime_root=tmp_path / "p4-acquisition", spec_path=spec
        )
