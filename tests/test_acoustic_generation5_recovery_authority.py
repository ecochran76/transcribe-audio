import json

import pytest

import acoustic_generation5_recovery_authority as r0


REPOSITORY = {
    "commit": "1" * 40,
    "module_sha256": {name: "2" * 64 for name in r0.MODULES},
    "clean": True,
    "upstream_ahead": 0,
    "upstream_behind": 0,
}
TERMINAL = {"content_sha256": r0.PLAN0053_STOP_PREVIEW_SHA256, "terminal_decision": "stop"}
EXCLUSIONS = {
    "json_file_count": 20,
    "json_file_set_sha256": "a" * 64,
    "excluded_hash_count": 100,
    "excluded_hash_set_sha256": "b" * 64,
}
TOOLS = {"ffprobe_path": "/usr/bin/ffprobe", "ffprobe_revision": "ffprobe test"}


def _inventory(count: int = 9) -> list[dict]:
    return [
        {
            "path": f"/private/{index}.m4a",
            "source_sha256": f"{index + 1:064x}",
            "transcript_path": f"/private/{index}.json",
            "transcript_sha256": f"{index + 100:064x}",
            "recording_start_original": f"2026-01-{index + 1:02d}T10:00:00-06:00",
            "recording_start_utc": f"2026-01-{index + 1:02d}T16:00:00Z",
            "status": "eligible",
            "reason_code": "eligible",
            "probe": {"codec_name": "aac"},
        }
        for index in range(count)
    ]


def _preview() -> dict:
    return r0.preview_generation5_recovery_authority(
        terminal_preview=TERMINAL,
        inventory=_inventory(),
        exclusion_summary=EXCLUSIONS,
        tool_identity=TOOLS,
        repository_authority=REPOSITORY,
    )


def test_r0_freezes_exact_roles_without_decode() -> None:
    preview = _preview()
    members = preview["private_evidence"]["selected_membership"]
    portable = r0._portable(preview)

    assert len(members) == 8
    assert members[0]["role"] == "recovery_negative_source"
    assert {item["role"] for item in members[1:]} == {"positive_holdout"}
    assert preview["did_decode_audio"] is False
    assert preview["action_vector"]["submit_exact_membership_to_j0"] is True
    assert preview["action_vector"]["decode_positive_holdout"] is False
    assert "private_evidence" not in portable
    assert "/private/" not in json.dumps(portable)


def test_r0_stops_when_fewer_than_eight_are_eligible() -> None:
    with pytest.raises(r0.Generation5RecoveryAuthorityError, match="insufficient"):
        r0.preview_generation5_recovery_authority(
            terminal_preview=TERMINAL,
            inventory=_inventory(7),
            exclusion_summary=EXCLUSIONS,
            tool_identity=TOOLS,
            repository_authority=REPOSITORY,
        )


def test_r0_apply_replay_is_private_and_idempotent(tmp_path, monkeypatch) -> None:
    preview = _preview()
    monkeypatch.setattr(r0, "preview_generation5_recovery_authority", lambda: preview)

    applied = r0.apply_generation5_recovery_authority(
        preview, expected_content_sha256=preview["content_sha256"], runtime_root=tmp_path
    )
    replayed = r0.replay_generation5_recovery_authority(
        preview["content_sha256"], runtime_root=tmp_path
    )

    paths = r0._paths(tmp_path, preview["content_sha256"])
    assert applied["idempotent_replay"] is False
    assert replayed["idempotent_replay"] is True
    assert paths["manifest"].stat().st_mode & 0o777 == 0o600
    assert "private_evidence" not in paths["receipt"].read_text()


def test_prior_reader_accepts_concatenated_json_documents(tmp_path) -> None:
    path = tmp_path / "evidence.json"
    path.write_text('{"one":"' + "a" * 64 + '"}\n{"two":"' + "b" * 64 + '"}\n')

    values = r0._read_json_sequence(path)

    assert len(values) == 2
    assert r0._all_hashes(values) == {"a" * 64, "b" * 64}


def test_prior_reader_conservatively_extracts_hashes_from_legacy_text(tmp_path) -> None:
    path = tmp_path / "legacy.json"
    path.write_text("legacy preface\n" + "c" * 64 + "\nnot-json\n")

    hashes, mode = r0._evidence_hashes(path)

    assert hashes == {"c" * 64}
    assert mode == "raw_sha256_token_fallback"
