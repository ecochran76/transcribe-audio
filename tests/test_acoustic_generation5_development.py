import json

import pytest

import acoustic_generation5_development as g1


REPOSITORY = {
    "commit": "1" * 40,
    "module_sha256": {name: "2" * 64 for name in g1.BOUND_MODULES},
    "clean": True,
    "upstream_ahead": 0,
    "upstream_behind": 0,
}


def _measurement(origin: str, status: str = "passing") -> dict:
    reasons = ["timeline_discontinuity"] if status == "rejected" else []
    return {
        "authority_origin": origin,
        "role": "known_failure" if "terminal" in origin else "healthy_control",
        "source_sha256": origin.encode().hex().ljust(64, "0")[:64],
        "measurement": {
            "status": status,
            "reason_codes": reasons,
            "output_sample_error": 0,
            "recipe_reference_decode": {"pcm_sha256": "a" * 64},
            "production_wav": {"pcm_sha256": "a" * 64},
        },
    }


def _measurements() -> list[dict]:
    return [
        _measurement("generation3_terminal_stop", "rejected"),
        _measurement("generation4_terminal_stop"),
        *[_measurement("plan0051_qualified_media") for _ in range(3)],
    ]


def _negative() -> dict:
    return {
        "case_count": 9,
        "all_expected_rejections_observed": True,
        "cases": [{"status": "rejected"}] * 9,
        "private_fixture_hashes": {"one": "b" * 64},
        "content_sha256": "c" * 64,
    }


def _preview() -> dict:
    return g1.preview_generation5_development(
        g0_preview={"content_sha256": g1.G0_PREVIEW_SHA256},
        measurements=_measurements(),
        adversarial_result=_negative(),
        repository_authority=REPOSITORY,
    )


def test_preview_freezes_diagnosis_without_authorizing_holdout() -> None:
    preview = _preview()
    portable = g1._portable(preview)

    assert preview["status"] == "ready_for_independent_j1_review"
    assert preview["diagnosis"]["generation4_content_loss_observed"] is False
    assert preview["action_vector"]["submit_to_j1"] is True
    assert preview["action_vector"]["measure_positive_holdout"] is False
    assert preview["did_measure_holdout"] is False
    assert "private_evidence" not in portable
    assert "private_fixture_hashes" not in json.dumps(portable)


def test_preview_rejects_case_failing_development_result() -> None:
    measurements = _measurements()
    measurements[-1]["measurement"]["status"] = "rejected"
    measurements[-1]["measurement"]["reason_codes"] = ["decode_warning"]

    with pytest.raises(g1.Generation5DevelopmentError, match="did not pass"):
        g1.preview_generation5_development(
            g0_preview={"content_sha256": g1.G0_PREVIEW_SHA256},
            measurements=measurements,
            adversarial_result=_negative(),
            repository_authority=REPOSITORY,
        )


def test_apply_and_replay_are_private_and_idempotent(tmp_path, monkeypatch) -> None:
    preview = _preview()
    monkeypatch.setattr(g1, "preview_generation5_development", lambda: preview)

    applied = g1.apply_generation5_development(
        preview, expected_content_sha256=preview["content_sha256"], runtime_root=tmp_path
    )
    replayed = g1.replay_generation5_development(
        preview["content_sha256"], runtime_root=tmp_path
    )

    paths = g1._paths(tmp_path, preview["content_sha256"])
    assert applied["idempotent_replay"] is False
    assert replayed["idempotent_replay"] is True
    assert paths["manifest"].stat().st_mode & 0o777 == 0o600
    assert "private_evidence" not in paths["receipt"].read_text()
