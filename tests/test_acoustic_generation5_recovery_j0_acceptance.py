import acoustic_generation5_recovery_j0_acceptance as j0


REPO = {"commit": "1" * 40, "module_sha256": "2" * 64, "clean": True, "upstream_ahead": 0, "upstream_behind": 0}
PARENT = {"content_sha256": j0.R0_PREVIEW_SHA256, "selected_membership_sha256": j0.SELECTED_MEMBERSHIP_SHA256, "did_decode_audio": False}


def _preview():
    return j0.preview_generation5_recovery_j0(r0_preview=PARENT, repository_authority=REPO)


def test_j0_authorizes_only_exact_r1_r2():
    preview = _preview()
    assert preview["review_decision"] == "PASS"
    assert preview["action_vector"]["run_exact_one_pass_r2"] is True
    assert preview["action_vector"]["enumerate_evaluation_candidates"] is False
    assert preview["did_decode_audio"] is False


def test_j0_private_apply_replay(tmp_path, monkeypatch):
    preview = _preview()
    monkeypatch.setattr(j0, "preview_generation5_recovery_j0", lambda: preview)
    applied = j0.apply_generation5_recovery_j0(preview, expected_content_sha256=preview["content_sha256"], runtime_root=tmp_path)
    replayed = j0.replay_generation5_recovery_j0(preview["content_sha256"], runtime_root=tmp_path)
    assert applied["idempotent_replay"] is False
    assert replayed["idempotent_replay"] is True
    assert j0._paths(tmp_path, preview["content_sha256"])["manifest"].stat().st_mode & 0o777 == 0o600
