import json

import pytest

import acoustic_generation5_diagnostic_authority as diagnostic


REPOSITORY = {
    "commit": "1" * 40,
    "module_name": diagnostic.MODULE_NAME,
    "module_sha256": "2" * 64,
    "clean": True,
    "upstream_ahead": 0,
    "upstream_behind": 0,
}


def _member(index: int, role: str, origin: str) -> dict:
    return {
        "source_sha256": f"{index:064x}",
        "path": f"/private/{index}.m4a",
        "role": role,
        "authority_origin": origin,
        "prior_reason_code": None,
        "prior_drift_seconds": None,
    }


def _bind(monkeypatch) -> None:
    monkeypatch.setattr(
        diagnostic,
        "_plan0052_authority",
        lambda: ({"private_evidence": {"cohort_membership": []}}, {"terminal_decision": "stop"}),
    )
    monkeypatch.setattr(
        diagnostic,
        "_generation3_failure",
        lambda: _member(1, "known_failure", "generation3_terminal_stop"),
    )
    monkeypatch.setattr(
        diagnostic,
        "_generation4_failure",
        lambda g2, terminal: _member(2, "known_failure", "generation4_terminal_stop"),
    )
    controls = [_member(index, "healthy_control", "plan0051_qualified_media") for index in range(3, 6)]
    holdout = [_member(index, "sealed_holdout", "plan0051_qualified_media") for index in range(6, 13)]
    monkeypatch.setattr(diagnostic, "_plan0051_split", lambda: (controls, holdout))
    monkeypatch.setattr(diagnostic, "_repository_authority", lambda: dict(REPOSITORY))


def test_preview_seals_disjoint_development_and_holdout(monkeypatch) -> None:
    _bind(monkeypatch)

    preview = diagnostic.preview_generation5_diagnostic_authority()
    portable = diagnostic._portable(preview)

    assert preview["status"] == "sealed_diagnostic_membership"
    assert preview["development_count"] == 5
    assert preview["known_failure_count"] == 2
    assert preview["healthy_control_count"] == 3
    assert preview["holdout_count"] == 7
    assert preview["action_vector"]["run_g1_development_diagnosis"] is True
    assert preview["action_vector"]["measure_holdout"] is False
    assert preview["did_decode_audio"] is False
    assert preview["did_measure_holdout"] is False
    assert "private_evidence" not in portable
    assert portable["contains_paths"] is False
    assert portable["contains_private_membership"] is False


def test_preview_rejects_membership_overlap(monkeypatch) -> None:
    _bind(monkeypatch)
    controls = [_member(index, "healthy_control", "plan0051_qualified_media") for index in range(3, 6)]
    holdout = [_member(index, "sealed_holdout", "plan0051_qualified_media") for index in range(5, 12)]
    monkeypatch.setattr(diagnostic, "_plan0051_split", lambda: (controls, holdout))

    with pytest.raises(diagnostic.Generation5DiagnosticAuthorityError, match="overlaps"):
        diagnostic.preview_generation5_diagnostic_authority()


def test_apply_replay_private_and_idempotent(tmp_path, monkeypatch) -> None:
    _bind(monkeypatch)
    preview = diagnostic.preview_generation5_diagnostic_authority()

    applied = diagnostic.apply_generation5_diagnostic_authority(
        preview,
        expected_content_sha256=preview["content_sha256"],
        runtime_root=tmp_path,
    )
    replayed = diagnostic.replay_generation5_diagnostic_authority(
        preview["content_sha256"], runtime_root=tmp_path
    )

    paths = diagnostic._paths(tmp_path, preview["content_sha256"])
    assert applied["idempotent_replay"] is False
    assert replayed["idempotent_replay"] is True
    assert paths["manifest"].stat().st_mode & 0o777 == 0o600
    assert paths["receipt"].stat().st_mode & 0o777 == 0o600
    assert "private_evidence" not in json.loads(paths["receipt"].read_text())
