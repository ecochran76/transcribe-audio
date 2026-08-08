from __future__ import annotations

from pathlib import Path

import pytest

import acoustic_plan0058 as plan


def _authority() -> dict:
    return {
        "commit": "a" * 40,
        "module_sha256": "b" * 64,
        "clean": True,
        "upstream_ahead": 0,
        "upstream_behind": 0,
    }


def test_preview_is_synthetic_complete_and_mutation_negative() -> None:
    preview = plan.preview_authority(repository_authority=_authority())

    assert preview["card_count"] == 15
    assert len(preview["expected_clip_sha256"]) == 15
    assert preview["contains_private_audio"] is False
    assert preview["contains_private_transcript"] is False
    assert preview["contains_private_identity_label"] is False
    assert set(preview["negative_actions"].values()) == {False}
    assert all("Synthetic" in card["transcript"] for card in preview["cards"])


def test_apply_and_replay_are_private_and_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority = _authority()
    monkeypatch.setattr(plan, "_repository_authority", lambda: authority)
    preview = plan.preview_authority(repository_authority=authority)

    applied = plan.apply_fixture(
        preview,
        expected_content_sha256=preview["content_sha256"],
        runtime_root=tmp_path,
    )
    replayed = plan.replay_fixture(
        preview["content_sha256"], runtime_root=tmp_path
    )
    applied_again = plan.apply_fixture(
        preview,
        expected_content_sha256=preview["content_sha256"],
        runtime_root=tmp_path,
    )
    run = Path(applied["fixture_path"])

    assert applied["idempotent_replay"] is False
    assert replayed["idempotent_replay"] is True
    assert applied_again["idempotent_replay"] is True
    assert replayed["index_sha256"] == preview["expected_index_sha256"]
    assert replayed["clip_sha256"] == preview["expected_clip_sha256"]
    assert run.stat().st_mode & 0o777 == 0o700
    for path in run.rglob("*"):
        assert path.stat().st_mode & 0o777 == (0o700 if path.is_dir() else 0o600)


def test_apply_rejects_stale_or_mutation_bearing_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority = _authority()
    monkeypatch.setattr(plan, "_repository_authority", lambda: authority)
    preview = plan.preview_authority(repository_authority=authority)
    preview["negative_actions"]["apply_speaker_assignments"] = True

    with pytest.raises(plan.Plan0058Error):
        plan.apply_fixture(
            preview,
            expected_content_sha256=preview["content_sha256"],
            runtime_root=tmp_path,
        )


def test_replay_rejects_clip_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority = _authority()
    monkeypatch.setattr(plan, "_repository_authority", lambda: authority)
    preview = plan.preview_authority(repository_authority=authority)
    applied = plan.apply_fixture(
        preview,
        expected_content_sha256=preview["content_sha256"],
        runtime_root=tmp_path,
    )
    clip = Path(applied["fixture_path"]) / "clips" / "card-15.wav"
    clip.chmod(0o600)
    clip.write_bytes(b"drift")

    with pytest.raises(plan.Plan0058Error):
        plan.replay_fixture(preview["content_sha256"], runtime_root=tmp_path)
