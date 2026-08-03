import json

import pytest

import acoustic_generation5_holdout as g2


REPOSITORY = {
    "commit": "1" * 40,
    "module_sha256": {name: "2" * 64 for name in g2.BOUND_MODULES},
    "clean": True,
    "upstream_ahead": 0,
    "upstream_behind": 0,
}
G0 = {"content_sha256": g2.G0_PREVIEW_SHA256}
J1 = {"content_sha256": g2.J1_PREVIEW_SHA256, "status": "accepted_for_g2_only"}


def _results() -> list[dict]:
    return [
        {
            "source_sha256": f"{index:064x}",
            "authority_origin": "plan0051_qualified_media",
            "measurement": {
                "status": "passing",
                "reason_codes": [],
                "output_sample_error": 0,
                "recipe_reference_decode": {"pcm_sha256": "a" * 64},
                "production_wav": {"pcm_sha256": "a" * 64},
            },
        }
        for index in range(7)
    ]


def _negative() -> dict:
    return {
        "seed": g2.adversarial.HOLDOUT_SEED,
        "case_count": 11,
        "all_expected_rejections_observed": True,
        "cases": [{"status": "rejected"}] * 11,
        "private_fixture_hashes": {"one": "b" * 64},
        "private_case_measurements": {"one": {"source_sha256": "c" * 64}},
        "content_sha256": "d" * 64,
    }


def _preview() -> dict:
    return g2.preview_generation5_holdout(
        g0_preview=G0,
        j1_preview=J1,
        holdout_results=_results(),
        heldout_adversarial=_negative(),
        repository_authority=REPOSITORY,
    )


def test_g2_passes_complete_positive_and_negative_denominators() -> None:
    preview = _preview()
    portable = g2._portable(preview)

    assert preview["positive_holdout_count"] == 7
    assert preview["positive_holdout_pass_count"] == 7
    assert preview["heldout_adversarial"]["case_count"] == 11
    assert preview["action_vector"]["submit_to_j2"] is True
    assert preview["action_vector"]["enumerate_generation5_candidates"] is False
    assert "private_evidence" not in portable
    assert "private_fixture_hashes" not in json.dumps(portable)


def test_g2_rejects_any_positive_holdout_failure() -> None:
    results = _results()
    results[-1]["measurement"]["status"] = "rejected"
    results[-1]["measurement"]["reason_codes"] = ["decode_warning"]

    with pytest.raises(g2.Generation5HoldoutError, match="did not pass"):
        g2.preview_generation5_holdout(
            g0_preview=G0, j1_preview=J1, holdout_results=results,
            heldout_adversarial=_negative(), repository_authority=REPOSITORY,
        )


def test_g2_apply_replay_is_private_and_idempotent(tmp_path, monkeypatch) -> None:
    preview = _preview()
    monkeypatch.setattr(g2, "preview_generation5_holdout", lambda: preview)

    applied = g2.apply_generation5_holdout(
        preview, expected_content_sha256=preview["content_sha256"], runtime_root=tmp_path
    )
    replayed = g2.replay_generation5_holdout(
        preview["content_sha256"], runtime_root=tmp_path
    )

    paths = g2._paths(tmp_path, preview["content_sha256"])
    assert applied["idempotent_replay"] is False
    assert replayed["idempotent_replay"] is True
    assert paths["manifest"].stat().st_mode & 0o777 == 0o600
    assert "private_evidence" not in paths["receipt"].read_text()
