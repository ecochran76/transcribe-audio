from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

import acoustic_generation2_authority as generation2
import acoustic_verification as verification
from acoustic_generation2_authority import Generation2AuthorityError


def _terminal_policy() -> dict:
    path = Path(__file__).parents[1] / (
        "docs/dev/fixtures/plan-0037-p4/"
        "generation-2-terminal-decision-policy.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _condition_inputs() -> tuple[dict, dict]:
    units = []
    splits = ["development"] * 3 + ["calibration"] * 2 + ["evaluation"] * 2
    for index, split in enumerate(splits, start=1):
        units.append(
            {
                "recording_id": f"recording-{index}",
                "conversation_id": f"conversation-{index}",
                "source_sha256": f"{index}" * 64,
                "split": split,
            }
        )
    manifest = {
        "schema_version": "transcribe-audio.acoustic-successor-condition-manifest.v1",
        "status": "complete",
        "content_sha256": generation2.EXPECTED_CONDITION_CONTENT_SHA256,
        "corpus": {
            "corpus_id": generation2.EXPECTED_SUCCESSOR_CORPUS_ID,
            "content_sha256": generation2.EXPECTED_SUCCESSOR_CORPUS_CONTENT_SHA256,
            "manifest_sha256": generation2.EXPECTED_SUCCESSOR_CORPUS_MANIFEST_SHA256,
        },
        "denominators": {
            "recordings": 7,
            "methods_per_recording": 5,
            "method_attempts": 35,
            "p1_successes": 7,
            "p2_method_successes": 35,
        },
        "units": units,
        "did_run_p1_p2": True,
        "did_process_audio": True,
        "did_read_private_corpus_gold_authority": True,
        "did_run_biometrics": False,
        "did_use_gold_for_condition_measurement": False,
        "did_perform_external_write": False,
        "contains_raw_audio": False,
        "contains_transcript_text": False,
        "contains_names_or_emails": False,
        "contains_embeddings_or_vectors": False,
    }
    replay = {
        "schema_version": "transcribe-audio.acoustic-successor-condition-replay.v1",
        "manifest_sha256": generation2.EXPECTED_CONDITION_MANIFEST_SHA256,
        "content_sha256": generation2.EXPECTED_CONDITION_CONTENT_SHA256,
        "full_body_match": True,
        "historical_authority_replay": True,
        "will_perform_external_write": False,
    }
    replay["safe_projection_sha256"] = generation2._canonical_hash(
        generation2._condition_safe_projection(manifest)
    )
    return manifest, replay


def _composite_inputs() -> tuple[dict, dict]:
    fields = {
        field: {
            "observed_values": [f"{field}-1", f"{field}-2"],
            "observed_value_count": 2,
            "missing_recordings": 0,
            "status": "pass",
        }
        for field in ("channel", "device", "noise", "telephone_bandwidth", "usable_duration_band")
    }
    coverage = {
        "fields": fields,
        "terminal_selection_eligible": True,
        "blockers": [],
    }
    core = {
        "schema_version": "transcribe-audio.acoustic-composite-condition-plan.v1",
        "campaign_id": "device-provenance-fixture",
        "campaign_manifest_sha256": "c" * 64,
        "campaign_records_state_sha256": "d" * 64,
        "condition_manifest_path": "/private/condition-manifest.json",
        "condition_manifest_sha256": generation2.EXPECTED_CONDITION_MANIFEST_SHA256,
        "condition_content_sha256": generation2.EXPECTED_CONDITION_CONTENT_SHA256,
        "recordings": 7,
        "latest_attestation_count": 7,
        "direct_observed_attestation_count": 7,
        "device_record_sha256": {
            f"recording-{index}": str(index) * 64 for index in range(1, 8)
        },
        "condition_coverage": coverage,
        "overlay_policy": {
            "only_device_may_change": True,
            "requires_seven_direct_observed": True,
            "minimum_distinct_devices": 2,
            "encoding_profile_is_not_device_evidence": True,
        },
        "will_run_models": False,
        "will_run_biometrics": False,
        "will_reveal_evaluation": False,
        "will_perform_external_write": False,
    }
    content_sha256 = generation2._canonical_hash(core)
    composite_id = f"composite-conditions-{content_sha256[:24]}"
    manifest = {
        **core,
        "schema_version": "transcribe-audio.acoustic-composite-condition-manifest.v1",
        "composite_id": composite_id,
        "content_sha256": content_sha256,
        "status": "complete",
        "applied_at": "2026-08-01T12:02:00Z",
    }
    replay = {
        "schema_version": "transcribe-audio.acoustic-composite-condition-receipt.v1",
        "composite_id": composite_id,
        "content_sha256": content_sha256,
        "manifest_sha256": "b" * 64,
        "condition_coverage": coverage,
        "full_body_match": True,
        "will_perform_external_write": False,
    }
    replay["safe_projection_sha256"] = generation2._canonical_hash(
        generation2._composite_safe_projection(manifest)
    )
    return manifest, replay


def _calibration_inputs() -> tuple[dict, dict]:
    candidates = (
        "speechbrain_ecapa_tdnn",
        "wespeaker_campplus",
        "wespeaker_resnet34",
    )
    methods = ["no_enhancement", "deepfilternet", "rnnoise"]
    profiles = []
    for candidate_index, candidate in enumerate(candidates, start=1):
        for person_index in (1, 2):
            profiles.append(
                {
                    "profile_id": f"profile-{candidate}-{person_index}",
                    "descendant_id": f"descendant-{candidate}-{person_index}",
                    "person_ref_id": f"person-{person_index}",
                    "candidate_id": candidate,
                    "model_revision": str(candidate_index) * 40,
                    "artifact_sha256": str(candidate_index) * 64,
                    "profile_manifest_sha256": str(candidate_index + 2) * 64,
                    "generation_sha256": str(candidate_index + 5) * 64,
                    "lifecycle_state": "active",
                }
            )
    thresholds = [
        {
            "candidate_id": candidate,
            "method_id": method,
            "status": "success",
            "threshold": 0.5,
            "temperature": 0.05,
        }
        for candidate in candidates
        for method in methods
    ]
    authority = {
        "status": "authorized",
        "intended_split": "calibration",
        "authorized_at": "2026-07-31T12:00:00Z",
        "profiles": profiles,
        "preparation_methods": [
            "no_enhancement",
            "silero_vad",
            "deepfilternet",
            "rnnoise",
            "pyannote_community_1",
        ],
        "score_methods": methods,
        "preparation_contract": {
            "channel_policy": {
                "allowed_source_channels": [1, 2],
                "mono": "identity",
                "stereo": "arithmetic_average_0.5_left_plus_0.5_right",
                "output_channels": 1,
                "authority_binding": "this_calibration_authority_sha256",
                "no_silent_fallback": True,
            }
        },
        "window_policy": {
            "minimum_seconds": 0.75,
            "maximum_seconds": 15.0,
            "maximum_windows_per_speaker_per_conversation": 3,
        },
        "metric_policy": {
            "condition_slices": [
                "channel",
                "device",
                "noise",
                "overlap",
                "telephone_bandwidth",
                "usable_duration_band",
            ]
        },
        "will_read_evaluation": False,
    }
    authority_sha256 = verification.canonical_artifact_hash(authority)
    authority = {**authority, "authority_sha256": authority_sha256}
    calibration = {
        "schema_version": verification.CALIBRATION_APPLICATION_SCHEMA,
        "status": "success",
        "intended_split": "calibration",
        "authority_sha256": authority_sha256,
        "score_matrix_sha256": "b" * 64,
        "threshold_unit_count": 9,
        "thresholds": thresholds,
        "did_select_and_freeze_thresholds": True,
        "did_read_evaluation": False,
        "did_make_terminal_model_or_method_selection": False,
        "permits_generalization_claim": False,
        "applied_at": "2026-07-31T12:01:00Z",
    }
    application_sha256 = verification._calibration_stage_identity(
        calibration, "applied_at"
    )
    calibration = {**calibration, "application_sha256": application_sha256}
    return calibration, authority


def test_legacy_composite_binding_shape_remains_stable() -> None:
    manifest, replay = _composite_inputs()
    binding = generation2._composite_binding(manifest, replay)
    assert set(binding) == {
        "composite_id",
        "content_sha256",
        "manifest_sha256",
        "condition_coverage",
        "direct_observed_attestation_count",
        "minimum_distinct_device_count",
    }
    assert binding["direct_observed_attestation_count"] == 7


@pytest.fixture(autouse=True)
def _synthetic_calibration_authority_constants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibration, authority = _calibration_inputs()
    monkeypatch.setattr(
        generation2,
        "EXPECTED_CALIBRATION_APPLICATION_SHA256",
        calibration["application_sha256"],
    )
    monkeypatch.setattr(
        generation2,
        "EXPECTED_CALIBRATION_SCORE_MATRIX_SHA256",
        calibration["score_matrix_sha256"],
    )
    monkeypatch.setattr(
        verification,
        "HISTORICAL_CALIBRATION_AUTHORITY_SHA256",
        authority["authority_sha256"],
    )
    condition_manifest, _ = _condition_inputs()
    monkeypatch.setattr(
        generation2,
        "EXPECTED_CONDITION_SAFE_PROJECTION_SHA256",
        generation2._canonical_hash(
            generation2._condition_safe_projection(condition_manifest)
        ),
    )


def _inputs() -> dict:
    condition_manifest, condition_replay = _condition_inputs()
    composite_manifest, composite_replay = _composite_inputs()
    calibration, calibration_authority = _calibration_inputs()
    return {
        "calibration": calibration,
        "calibration_authority": calibration_authority,
        "condition_manifest": condition_manifest,
        "condition_replay": condition_replay,
        "composite_manifest": composite_manifest,
        "composite_replay": composite_replay,
        "terminal_policy": _terminal_policy(),
        "terminal_policy_sha256": (
            generation2.EXPECTED_GENERATION2_TERMINAL_POLICY_SHA256
        ),
    }


def _historical_condition_fixture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[Path, Path, Path, Path]:
    root = tmp_path / "condition-runtime"
    run = root / "runs" / generation2.EXPECTED_CONDITION_PLAN_ID
    artifacts = root / "artifacts"
    run.mkdir(parents=True)
    artifacts.mkdir()
    for directory in (root, root / "runs", run, artifacts):
        directory.chmod(0o700)
    corpus_path = root / "corpus.json"
    corpus_path.write_text("{}", encoding="utf-8")
    corpus_path.chmod(0o600)
    units = []
    p1_to_tamper = artifacts / "p1-audio-1.wav"
    p2_to_tamper = artifacts / "output-7-pyannote_community_1.wav"
    for index in range(1, 8):
        p1_audio = artifacts / f"p1-audio-{index}.wav"
        p1_audio.write_bytes(f"p1-audio-{index}".encode())
        p1_audio.chmod(0o600)
        p1_audio_sha = hashlib.sha256(p1_audio.read_bytes()).hexdigest()
        methods = []
        for method_id in generation2.successor_conditions.METHOD_IDS:
            output = artifacts / f"output-{index}-{method_id}.wav"
            output.write_bytes(f"output-{index}-{method_id}".encode())
            output.chmod(0o600)
            methods.append(
                {
                    "method_id": method_id,
                    "status": "success",
                    "output_path": str(output),
                    "output_sha256": hashlib.sha256(
                        output.read_bytes()
                    ).hexdigest(),
                }
            )
        p1 = artifacts / f"p1-{index}.json"
        p2 = artifacts / f"p2-{index}.json"
        p1_replay = artifacts / f"p1-replay-{index}.json"
        p2_replay = artifacts / f"p2-replay-{index}.json"
        p1.write_text(
            json.dumps(
                {
                    "index": index,
                    "artifact_path": str(p1_audio),
                    "derived_audio": {"output_sha256": p1_audio_sha},
                }
            ),
            encoding="utf-8",
        )
        p2.write_text(json.dumps({"method_results": methods}), encoding="utf-8")
        p1_replay.write_text("{}", encoding="utf-8")
        p2_replay.write_text("{}", encoding="utf-8")
        for path in (p1, p2, p1_replay, p2_replay):
            path.chmod(0o600)
        digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        units.append(
            {
                "recording_id": f"recording-{index}",
                "conversation_id": f"conversation-{index}",
                "source_sha256": f"{index}" * 64,
                "split": (
                    "development" if index <= 3
                    else "calibration" if index <= 5
                    else "evaluation"
                ),
                "p1_manifest_path": str(p1),
                "p1_manifest_sha256": digest(p1),
                "p2_comparison_path": str(p2),
                "p2_comparison_sha256": digest(p2),
                "p1_replay_path": str(p1_replay),
                "p1_replay_sha256": digest(p1_replay),
                "p2_replay_path": str(p2_replay),
                "p2_replay_sha256": digest(p2_replay),
                "method_result_sha256": {
                    method["method_id"]: generation2._canonical_hash(method)
                    for method in methods
                },
                "conditions": {"index": index},
            }
        )
    coverage = {"fixture": "pass"}
    denominators = {
        "recordings": 7,
        "methods_per_recording": 5,
        "method_attempts": 35,
        "p1_successes": 7,
        "p2_method_successes": 35,
    }
    manifest = {
        "schema_version": "transcribe-audio.acoustic-successor-condition-manifest.v1",
        "status": "complete",
        "plan_id": generation2.EXPECTED_CONDITION_PLAN_ID,
        "plan_content_sha256": generation2.EXPECTED_CONDITION_PLAN_CONTENT_SHA256,
        "content_sha256": generation2.EXPECTED_CONDITION_CONTENT_SHA256,
        "repository_authority": {
            "clean": True,
            "commit": generation2.EXPECTED_CONDITION_REPOSITORY_COMMIT,
            "module_sha256": generation2.EXPECTED_CONDITION_MODULE_SHA256,
        },
        "module_authority": {
            "condition_sha256": generation2.EXPECTED_CONDITION_MODULE_SHA256,
            "p1_sha256": generation2.EXPECTED_CONDITION_P1_MODULE_SHA256,
            "p2_sha256": generation2.EXPECTED_CONDITION_P2_MODULE_SHA256,
        },
        "readiness_sha256": generation2.EXPECTED_CONDITION_READINESS_SHA256,
        "corpus": {
            "manifest_path": str(corpus_path),
            "manifest_sha256": generation2.EXPECTED_SUCCESSOR_CORPUS_MANIFEST_SHA256,
        },
        "denominators": denominators,
        "condition_coverage": coverage,
        "units": units,
        "applied_at": "2026-08-01T12:00:00Z",
    }
    manifest_path = run / "condition-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_path.chmod(0o600)
    receipt = {
        "schema_version": "transcribe-audio.acoustic-successor-condition-receipt.v1",
        "plan_id": generation2.EXPECTED_CONDITION_PLAN_ID,
        "manifest_path": str(manifest_path),
        "manifest_sha256": generation2.EXPECTED_CONDITION_MANIFEST_SHA256,
        "content_sha256": generation2.EXPECTED_CONDITION_CONTENT_SHA256,
        "denominators": denominators,
        "condition_coverage": coverage,
        "mode": "0600",
        "will_perform_external_write": False,
    }
    receipt_path = run / "apply-receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    receipt_path.chmod(0o600)

    original_hash = generation2._canonical_hash

    def canonical(value):
        if (
            isinstance(value, dict)
            and value.get("schema_version")
            == "transcribe-audio.acoustic-successor-condition-manifest.v1"
        ):
            return generation2.EXPECTED_CONDITION_CONTENT_SHA256
        return original_hash(value)

    def file_hash(path):
        selected = Path(path).resolve()
        fixed = {
            manifest_path.resolve(): generation2.EXPECTED_CONDITION_MANIFEST_SHA256,
            corpus_path.resolve(): generation2.EXPECTED_SUCCESSOR_CORPUS_MANIFEST_SHA256,
            Path(generation2.successor_conditions.__file__).resolve(): (
                generation2.EXPECTED_CONDITION_MODULE_SHA256
            ),
            Path(generation2.audio_derivatives.__file__).resolve(): (
                generation2.EXPECTED_CONDITION_P1_MODULE_SHA256
            ),
            Path(generation2.speech_preparation.__file__).resolve(): (
                generation2.EXPECTED_CONDITION_P2_MODULE_SHA256
            ),
        }
        return fixed.get(selected, hashlib.sha256(selected.read_bytes()).hexdigest())

    monkeypatch.setattr(generation2, "_canonical_hash", canonical)
    monkeypatch.setattr(generation2.audio_derivatives, "sha256_file", file_hash)
    monkeypatch.setattr(
        generation2.successor_conditions,
        "_conditions",
        lambda p1, p2: {"index": p1["index"]},
    )
    monkeypatch.setattr(
        generation2.successor_conditions,
        "_aggregate_conditions",
        lambda selected_units: coverage,
    )
    return manifest_path, corpus_path, p1_to_tamper, p2_to_tamper


def test_preview_is_deterministic_replayable_and_cannot_score() -> None:
    first = generation2.preview_generation2_pre_reveal_authority(**_inputs())
    second = generation2.preview_generation2_pre_reveal_authority(**_inputs())
    replay = generation2.replay_generation2_pre_reveal_preview(first, **_inputs())

    assert first == second
    assert replay["full_body_match"] is True
    assert first["authority_generation"] == 2
    assert first["successor_seal"]["split_counts"] == {
        "development": 3,
        "calibration": 2,
        "evaluation": 2,
    }
    assert len(first["successor_seal"]["evaluation_records"]) == 2
    assert len(first["candidate_matrix"]) == 9
    assert len(first["frozen_thresholds"]) == 9
    assert all(item["margin"] == 0.0 for item in first["fixed_abstention_margins"])
    assert first["will_reveal_evaluation_after_apply"] is True
    assert first["production_apply_authorized"] is False
    assert first["requires_independent_review"] is True
    for field in (
        "will_run_models",
        "will_score_trials",
        "will_calculate_terminal_metrics",
        "will_make_terminal_decision",
        "will_perform_external_write",
    ):
        assert first[field] is False
    child = first["exact_trial_child_policy"]
    assert child["required_before_model_or_score_execution"] is True
    assert child["must_freeze_exact_trial_ids"] is True
    assert child["may_change_parent_policy_threshold_margin_or_candidate"] is False


def test_historical_condition_replay_is_read_only_and_exact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest_path, corpus_path, _, _ = _historical_condition_fixture(
        monkeypatch, tmp_path
    )
    monkeypatch.setattr(
        generation2.audio_derivatives,
        "write_immutable_private_json",
        lambda *args, **kwargs: pytest.fail("writer entered historical replay"),
        raising=False,
    )
    replay = generation2.replay_historical_condition_campaign(
        manifest_path, corpus_manifest_path=corpus_path
    )
    assert replay["full_body_match"] is True
    assert replay["historical_authority_replay"] is True
    assert replay["will_perform_external_write"] is False


def test_historical_condition_replay_rejects_lineage_output_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest_path, corpus_path, _, output = _historical_condition_fixture(
        monkeypatch, tmp_path
    )
    output.write_bytes(b"tampered")
    with pytest.raises(Generation2AuthorityError, match="method output drifted"):
        generation2.replay_historical_condition_campaign(
            manifest_path, corpus_manifest_path=corpus_path
        )


def test_historical_condition_replay_rejects_p1_audio_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest_path, corpus_path, output, _ = _historical_condition_fixture(
        monkeypatch, tmp_path
    )
    output.write_bytes(b"tampered")
    with pytest.raises(Generation2AuthorityError, match="P1 audio artifact drifted"):
        generation2.replay_historical_condition_campaign(
            manifest_path, corpus_manifest_path=corpus_path
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "condition_hash",
        "condition_full_body",
        "condition_projection_forge",
        "split_count",
        "duplicate_recording",
        "composite_blocked",
        "composite_projection_forge",
        "device_missing",
        "calibration_hash",
        "threshold_missing",
        "threshold_value",
        "profile_inactive",
        "profile_revision",
        "profile_artifact",
        "terminal_policy_hash",
        "terminal_minimum",
    ],
)
def test_preview_rejects_every_predecessor_or_policy_drift(mutation: str) -> None:
    inputs = _inputs()
    if mutation == "condition_hash":
        inputs["condition_replay"]["manifest_sha256"] = "0" * 64
    elif mutation == "condition_full_body":
        inputs["condition_replay"]["full_body_match"] = False
    elif mutation == "condition_projection_forge":
        inputs["condition_manifest"]["units"][0]["source_sha256"] = "f" * 64
        inputs["condition_replay"]["safe_projection_sha256"] = (
            generation2._canonical_hash(
                generation2._condition_safe_projection(
                    inputs["condition_manifest"]
                )
            )
        )
    elif mutation == "split_count":
        inputs["condition_manifest"]["units"][0]["split"] = "evaluation"
    elif mutation == "duplicate_recording":
        inputs["condition_manifest"]["units"][1]["recording_id"] = "recording-1"
    elif mutation == "composite_blocked":
        inputs["composite_replay"]["condition_coverage"][
            "terminal_selection_eligible"
        ] = False
    elif mutation == "composite_projection_forge":
        inputs["composite_replay"]["condition_coverage"]["fields"]["device"][
            "observed_values"
        ] = ["forged-device-1", "forged-device-2"]
        inputs["composite_replay"]["safe_projection_sha256"] = (
            generation2._canonical_hash(
                generation2._composite_safe_projection(
                    inputs["composite_manifest"]
                )
            )
        )
    elif mutation == "device_missing":
        inputs["composite_replay"]["condition_coverage"]["fields"]["device"][
            "missing_recordings"
        ] = 1
    elif mutation == "calibration_hash":
        inputs["calibration"]["score_matrix_sha256"] = "0" * 64
    elif mutation == "threshold_missing":
        inputs["calibration"]["thresholds"].pop()
    elif mutation == "threshold_value":
        inputs["calibration"]["thresholds"][0]["threshold"] = 0.75
    elif mutation == "profile_inactive":
        inputs["calibration_authority"]["profiles"][0]["lifecycle_state"] = "withdrawn"
    elif mutation == "profile_revision":
        inputs["calibration_authority"]["profiles"][0]["model_revision"] = "f" * 40
    elif mutation == "profile_artifact":
        inputs["calibration_authority"]["profiles"][0]["artifact_sha256"] = "f" * 64
    elif mutation == "terminal_policy_hash":
        inputs["terminal_policy_sha256"] = "0" * 64
    else:
        inputs["terminal_policy"]["minimum_evidence"][
            "evaluation_recordings"
        ] = 1

    with pytest.raises(Generation2AuthorityError):
        generation2.preview_generation2_pre_reveal_authority(**inputs)


def test_preview_replay_rejects_any_stored_body_change() -> None:
    preview = generation2.preview_generation2_pre_reveal_authority(**_inputs())
    forged = copy.deepcopy(preview)
    forged["will_score_trials"] = True
    forged["content_sha256"] = generation2._canonical_hash(
        {key: value for key, value in forged.items() if key not in {"preview_id", "content_sha256"}}
    )
    with pytest.raises(Generation2AuthorityError, match="replay mismatch"):
        generation2.replay_generation2_pre_reveal_preview(forged, **_inputs())


def test_preview_rejects_forbidden_private_fields() -> None:
    inputs = _inputs()
    inputs["composite_replay"]["condition_coverage"]["name"] = "private"
    inputs["composite_replay"]["safe_projection_sha256"] = (
        generation2._canonical_hash(
            generation2._composite_safe_projection(inputs["composite_manifest"])
        )
    )
    with pytest.raises(Generation2AuthorityError):
        generation2.preview_generation2_pre_reveal_authority(**inputs)
