import hashlib

import acoustic_generation5_recovery_j2_acceptance as j2


REPO = {
    "commit": "1" * 40,
    "module_sha256": hashlib.sha256(b"module").hexdigest(),
    "clean": True,
    "upstream_ahead": 0,
    "upstream_behind": 0,
}
PARENT = {"content_sha256": j2.R2_PREVIEW_SHA256, "positive_holdout_pass_count": 7, "recovery_negative": {"all_expected_rejections_observed": True}}


def _preview():
    return j2.preview_generation5_recovery_j2(r2_preview=PARENT, repository_authority=REPO)


def test_j2_authorizes_e1_only():
    preview = _preview()
    assert preview["review_decision"] == "PASS"
    assert preview["action_vector"]["enumerate_e1_candidates"] is True
    assert preview["action_vector"]["run_models_or_predictions"] is False


def test_j2_apply_replay(tmp_path, monkeypatch):
    preview = _preview()
    monkeypatch.setattr(j2, "preview_generation5_recovery_j2", lambda: preview)
    monkeypatch.setattr(j2, "_r2_preview", lambda: PARENT)
    monkeypatch.setattr(
        j2,
        "_git",
        lambda arguments, binary=False: b"module" if arguments[0] == "show" else "",
    )
    applied = j2.apply_generation5_recovery_j2(
        preview,
        expected_content_sha256=preview["content_sha256"],
        runtime_root=tmp_path,
    )
    replayed = j2.replay_generation5_recovery_j2(preview["content_sha256"], runtime_root=tmp_path)
    assert applied["idempotent_replay"] is False
    assert replayed["idempotent_replay"] is True
    assert j2._paths(tmp_path, preview["content_sha256"])["manifest"].stat().st_mode & 0o777 == 0o600
