import json

import pytest

import acoustic_generation5_j1_acceptance as j1


REPOSITORY = {
    "commit": "1" * 40,
    "module_sha256": "2" * 64,
    "clean": True,
    "upstream_ahead": 0,
    "upstream_behind": 0,
}


PARENT = {
    "content_sha256": j1.G1_PREVIEW_SHA256,
    "contract_sha256": j1.G1_CONTRACT_SHA256,
    "status": "ready_for_independent_j1_review",
    "did_measure_holdout": False,
}


def _preview() -> dict:
    return j1.preview_generation5_j1_acceptance(
        g1_preview=PARENT, repository_authority=REPOSITORY
    )


def test_acceptance_authorizes_g2_only() -> None:
    preview = _preview()

    assert preview["review_decision"] == "PASS"
    assert preview["status"] == "accepted_for_g2_only"
    assert preview["action_vector"]["run_g2_positive_holdout_once"] is True
    assert preview["action_vector"]["instantiate_g2_heldout_negative_family_once"] is True
    assert preview["action_vector"]["enumerate_generation5_candidates"] is False
    assert preview["did_measure_holdout"] is False


def test_acceptance_rejects_parent_drift() -> None:
    with pytest.raises(j1.Generation5J1AcceptanceError, match="invalid"):
        j1.preview_generation5_j1_acceptance(
            g1_preview={**PARENT, "did_measure_holdout": True},
            repository_authority=REPOSITORY,
        )


def test_apply_replay_are_private_and_idempotent(tmp_path, monkeypatch) -> None:
    preview = _preview()
    monkeypatch.setattr(j1, "preview_generation5_j1_acceptance", lambda: preview)

    applied = j1.apply_generation5_j1_acceptance(
        preview, expected_content_sha256=preview["content_sha256"], runtime_root=tmp_path
    )
    replayed = j1.replay_generation5_j1_acceptance(
        preview["content_sha256"], runtime_root=tmp_path
    )

    paths = j1._paths(tmp_path, preview["content_sha256"])
    assert applied["idempotent_replay"] is False
    assert replayed["idempotent_replay"] is True
    assert paths["manifest"].stat().st_mode & 0o777 == 0o600
    receipt = json.loads(paths["receipt"].read_text())
    assert "repository_authority" not in receipt
