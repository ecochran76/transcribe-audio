from __future__ import annotations

from pathlib import Path

import pytest

import speaker_identity_plan0068_a0 as a0


def test_legacy_binding_preserves_observed_mode_and_hash(tmp_path: Path) -> None:
    artifact = tmp_path / "legacy.json"
    artifact.write_text("{}", encoding="utf-8")
    artifact.chmod(0o644)

    binding = a0.legacy_input_binding(artifact, tmp_path)

    assert binding["observed_mode"] == "0644"
    assert binding["mode_was_changed"] is False
    assert binding["file_sha256"]
    assert artifact.stat().st_mode & 0o777 == 0o644


def test_legacy_binding_rejects_symlink_and_outside_root(tmp_path: Path) -> None:
    artifact = tmp_path / "legacy.json"
    artifact.write_text("{}", encoding="utf-8")
    symlink = tmp_path / "link.json"
    symlink.symlink_to(artifact)

    with pytest.raises(a0.Plan0068A0Error, match="non-symlinked"):
        a0.legacy_input_binding(symlink, tmp_path)

    outside = tmp_path.parent / "outside-plan0068.json"
    outside.write_text("{}", encoding="utf-8")
    try:
        with pytest.raises(a0.Plan0068A0Error, match="outside its authority root"):
            a0.legacy_input_binding(outside, tmp_path)
    finally:
        outside.unlink()
