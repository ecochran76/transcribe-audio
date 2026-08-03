import json
import hashlib

import pytest

import acoustic_generation5_j2_stop as stop


REPOSITORY = {
    "commit": "1" * 40,
    "module_sha256": hashlib.sha256(b"module").hexdigest(),
    "clean": True,
    "upstream_ahead": 0,
    "upstream_behind": 0,
}
G2 = {
    "content_sha256": stop.G2_PREVIEW_SHA256,
    "positive_holdout_count": 7,
    "positive_holdout_pass_count": 7,
    "heldout_adversarial": {"case_count": 11},
}


def _preview() -> dict:
    return stop.preview_generation5_j2_stop(
        g2_preview=G2, repository_authority=REPOSITORY
    )


def test_j2_stop_is_terminal_and_has_no_positive_actions() -> None:
    preview = _preview()

    assert preview["terminal_decision"] == "stop"
    assert preview["terminal_stage"] == "J2_independent_validation_audit"
    assert preview["finding"]["fixed_negative_cases_reproduced"] == 10
    assert preview["finding"]["required_negative_cases"] == 11
    assert not any(preview["action_vector"].values())


def test_j2_stop_rejects_parent_drift() -> None:
    with pytest.raises(stop.Generation5J2StopError, match="drifted"):
        stop.preview_generation5_j2_stop(
            g2_preview={**G2, "content_sha256": "a" * 64},
            repository_authority=REPOSITORY,
        )


def test_j2_stop_apply_replay_is_private_and_idempotent(tmp_path, monkeypatch) -> None:
    preview = _preview()
    monkeypatch.setattr(stop, "preview_generation5_j2_stop", lambda: preview)
    monkeypatch.setattr(stop, "_g2_preview", lambda: G2)
    monkeypatch.setattr(
        stop,
        "_git",
        lambda arguments, binary=False: b"module" if arguments[0] == "show" else "",
    )

    applied = stop.apply_generation5_j2_stop(
        preview, expected_content_sha256=preview["content_sha256"], runtime_root=tmp_path
    )
    replayed = stop.replay_generation5_j2_stop(
        preview["content_sha256"], runtime_root=tmp_path
    )

    paths = stop._paths(tmp_path, preview["content_sha256"])
    assert applied["idempotent_replay"] is False
    assert replayed["idempotent_replay"] is True
    assert paths["manifest"].stat().st_mode & 0o777 == 0o600
    assert "repository_authority" not in json.loads(paths["receipt"].read_text())
