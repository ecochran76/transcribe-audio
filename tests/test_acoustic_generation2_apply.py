from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import acoustic_generation2_apply as apply2


def _preview() -> dict:
    return {
        "schema_version": "fixture.preview.v1",
        "preview_id": "generation-2-pre-reveal-fixture",
        "content_sha256": "a" * 64,
        "status": "ready_for_independent_review",
        "production_apply_authorized": False,
        "will_run_models": False,
        "will_score_trials": False,
        "will_perform_external_write": False,
    }


def _repo() -> dict:
    return {
        "commit": "a" * 40,
        "generation2_module_sha256": "b" * 64,
        "apply_module_sha256": "c" * 64,
        "clean": True,
        "upstream_ahead": 0,
        "upstream_behind": 0,
    }


def test_apply_replay_is_private_idempotent_and_never_runs_models(tmp_path, monkeypatch):
    preview = _preview()
    monkeypatch.setattr(
        apply2.generation2,
        "replay_generation2_pre_reveal_preview",
        lambda stored, **inputs: {
            "content_sha256": stored["content_sha256"],
            "full_body_match": True,
        },
    )
    monkeypatch.setattr(apply2, "_repository_authority", _repo)
    monkeypatch.setattr(apply2, "_validate_repository_authority", lambda frozen: None)
    applied = apply2.apply_generation2_pre_reveal(
        preview,
        expected_preview_content_sha256=preview["content_sha256"],
        preview_inputs={},
        runtime_root=tmp_path / "runtime",
    )
    receipt = json.loads(Path(applied["receipt_path"]).read_text(encoding="utf-8"))
    assert receipt["evaluation_reveal_authorized"] is True
    assert receipt["model_execution_authorized"] is False
    assert receipt["trial_scoring_authorized"] is False
    assert set(receipt) == {
        "schema_version", "authority_id", "authority_content_sha256", "preview_id",
        "preview_content_sha256", "manifest_sha256", "evaluation_reveal_authorized",
        "model_execution_authorized", "trial_scoring_authorized",
        "contains_private_evaluation", "contains_device_labels", "mode",
        "will_perform_external_write",
    }
    replay = apply2.replay_generation2_pre_reveal(
        Path(applied["manifest_path"]), preview_inputs={}, runtime_root=tmp_path / "runtime"
    )
    assert replay["full_body_match"] is True
    monkeypatch.setattr(
        apply2,
        "_repository_authority",
        lambda: {**_repo(), "commit": "d" * 40},
    )
    assert apply2.apply_generation2_pre_reveal(
        preview,
        expected_preview_content_sha256=preview["content_sha256"],
        preview_inputs={},
        runtime_root=tmp_path / "runtime",
    )["idempotent"] is True
    manifest_path = Path(applied["manifest_path"])
    receipt_path = Path(applied["receipt_path"])
    forged = json.loads(manifest_path.read_text(encoding="utf-8"))
    forged["model_execution_authorized"] = True
    supplied_core = {
        key: value
        for key, value in forged.items()
        if key not in {"authority_id", "content_sha256"}
    }
    forged_sha = apply2._canonical_hash(supplied_core)
    forged["content_sha256"] = forged_sha
    forged["authority_id"] = f"generation-2-pre-reveal-authority-{forged_sha[:24]}"
    manifest_path.write_text(json.dumps(forged, sort_keys=True), encoding="utf-8")
    manifest_path.chmod(0o600)
    forged_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    forged_receipt["authority_id"] = forged["authority_id"]
    forged_receipt["authority_content_sha256"] = forged_sha
    forged_receipt["manifest_sha256"] = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    receipt_path.write_text(json.dumps(forged_receipt, sort_keys=True), encoding="utf-8")
    receipt_path.chmod(0o600)
    with pytest.raises(apply2.Generation2ApplyError, match="full-body"):
        apply2.replay_generation2_pre_reveal(
            manifest_path, preview_inputs={}, runtime_root=tmp_path / "runtime"
        )


def test_apply_rejects_stale_reviewed_hash(tmp_path, monkeypatch):
    preview = _preview()
    monkeypatch.setattr(
        apply2.generation2,
        "replay_generation2_pre_reveal_preview",
        lambda stored, **inputs: {"content_sha256": stored["content_sha256"]},
    )
    with pytest.raises(apply2.Generation2ApplyError, match="stale or unsafe"):
        apply2.apply_generation2_pre_reveal(
            preview,
            expected_preview_content_sha256="f" * 64,
            preview_inputs={},
            runtime_root=tmp_path / "runtime",
        )


def test_apply_rejects_receipt_only_partial_authority(tmp_path, monkeypatch):
    preview = _preview()
    root = tmp_path / "runtime"
    partial = root / "authorities" / "partial"
    partial.mkdir(parents=True)
    for directory in (root, root / "authorities", partial):
        directory.chmod(0o700)
    receipt = partial / "apply-receipt.json"
    receipt.write_text("{}", encoding="utf-8")
    receipt.chmod(0o600)
    monkeypatch.setattr(
        apply2.generation2,
        "replay_generation2_pre_reveal_preview",
        lambda stored, **inputs: {"content_sha256": stored["content_sha256"]},
    )
    monkeypatch.setattr(apply2, "_repository_authority", _repo)
    with pytest.raises(apply2.Generation2ApplyError, match="Partial or unknown"):
        apply2.apply_generation2_pre_reveal(
            preview,
            expected_preview_content_sha256=preview["content_sha256"],
            preview_inputs={},
            runtime_root=root,
        )
