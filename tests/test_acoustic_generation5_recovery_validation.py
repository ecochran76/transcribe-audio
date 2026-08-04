import acoustic_generation5_recovery_validation as r2


REPO = {"commit": "1" * 40, "module_sha256": {}, "clean": True, "upstream_ahead": 0, "upstream_behind": 0}
R0 = {"content_sha256": r2.R0_PREVIEW_SHA256}
J0 = {"content_sha256": r2.J0_PREVIEW_SHA256, "status": "accepted_for_exact_r1_r2_only"}


def _positive():
    return [{"source_sha256": f"{i:064x}", "ordinal": i + 2, "measurement": {"status": "passing", "reason_codes": [], "output_sample_error": 0, "recipe_reference_decode": {"pcm_sha256": "a" * 64}, "production_wav": {"pcm_sha256": "a" * 64}}} for i in range(7)]


def _negative():
    cases = []
    names = [f"tail_loss_{i}" for i in range(4)] + ["middle", "compressed", "corrupt_output", "timestamp_1", "timestamp_2", "wrong", "corrupt_source"]
    reasons = list(r2.adversarial.EXPECTED_REASON_CONTRACT.values())
    for i, name in enumerate(names):
        cases.append({"case_id": name, "status": "rejected", "expected_reason": reasons[min(i, len(reasons)-1)], "expected_reason_observed": True})
    return {"seed": r2.adversarial.RECOVERY_HOLDOUT_SEED, "case_count": 11, "expected_reason_contract": r2.adversarial.EXPECTED_REASON_CONTRACT, "expected_reason_contract_sha256": r2._canonical_hash(r2.adversarial.EXPECTED_REASON_CONTRACT), "all_expected_rejections_observed": True, "cases": cases, "private_fixture_hashes": {}, "private_case_measurements": {}}


def _preview():
    return r2.preview_generation5_recovery_validation(r0_preview=R0, j0_preview=J0, positive_results=_positive(), negative_result=_negative(), repository_authority=REPO)


def test_r2_complete_denominators_and_no_candidate_authority():
    preview = _preview()
    assert preview["positive_holdout_pass_count"] == 7
    assert preview["recovery_negative"]["case_count"] == 11
    assert preview["action_vector"]["submit_to_independent_j2"] is True
    assert preview["action_vector"]["enumerate_evaluation_candidates"] is False


def test_r2_apply_freezes_reviewed_body_without_second_execution(tmp_path, monkeypatch):
    preview = _preview()
    monkeypatch.setattr(r2, "_parents", lambda: (R0, J0))
    monkeypatch.setattr(r2, "_repository_authority", lambda: REPO)
    applied = r2.apply_generation5_recovery_validation(preview, expected_content_sha256=preview["content_sha256"], runtime_root=tmp_path)
    replayed = r2.replay_generation5_recovery_validation(preview["content_sha256"], runtime_root=tmp_path)
    assert applied["idempotent_replay"] is False
    assert replayed["idempotent_replay"] is True
