from __future__ import annotations

import hashlib
from pathlib import Path

import acoustic_plan0056_execution as execution


SUBJECT_IDS = (
    "subject-7c24e8f41409c6f517291fe7",
    "subject-df34bc192c07bd86566fff12",
)
CANDIDATES = (
    "speechbrain_ecapa_tdnn",
    "wespeaker_campplus",
    "wespeaker_resnet34",
)
METHODS = ("no_enhancement", "deepfilternet", "rnnoise")


def matrices(*, supporting_units: int, opposing_units: int = 0) -> list[dict]:
    values = []
    for index, (candidate, method) in enumerate(
        (pair for candidate in CANDIDATES for pair in ((candidate, item) for item in METHODS))
    ):
        if index < supporting_units:
            scores = [
                {"subject_id": SUBJECT_IDS[0], "score": 0.8},
                {"subject_id": SUBJECT_IDS[1], "score": 0.2},
            ]
        elif index < supporting_units + opposing_units:
            scores = [
                {"subject_id": SUBJECT_IDS[0], "score": 0.2},
                {"subject_id": SUBJECT_IDS[1], "score": 0.8},
            ]
        else:
            scores = [
                {"subject_id": SUBJECT_IDS[0], "score": 0.2},
                {"subject_id": SUBJECT_IDS[1], "score": 0.1},
            ]
        values.append(
            {
                "candidate_id": candidate,
                "method_id": method,
                "threshold": 0.5,
                "rows": [{"speaker_ref": "recording-1 / SPEAKER_1", "scores": scores}],
            }
        )
    return values


def test_consensus_assigns_only_after_six_units_and_two_families() -> None:
    proposals = execution.proposals_from_matrices(
        matrices(supporting_units=6),
        expected_speaker_refs=("recording-1 / SPEAKER_1",),
        allowlisted_subject_ids=SUBJECT_IDS,
    )

    proposal = proposals["proposals"][0]
    assert proposal["disposition"] == "assign"
    assert proposal["subject_id"] == SUBJECT_IDS[0]
    assert proposal["confidence_band"] == "medium"
    assert proposal["supporting_unit_count"] == 6
    assert proposal["opposing_unit_count"] == 0


def test_consensus_reviews_opposition_and_abstains_without_support() -> None:
    review = execution.proposals_from_matrices(
        matrices(supporting_units=6, opposing_units=1),
        expected_speaker_refs=("recording-1 / SPEAKER_1",),
        allowlisted_subject_ids=SUBJECT_IDS,
    )["proposals"][0]
    abstain = execution.proposals_from_matrices(
        matrices(supporting_units=0),
        expected_speaker_refs=("recording-1 / SPEAKER_1",),
        allowlisted_subject_ids=SUBJECT_IDS,
    )["proposals"][0]

    assert review["disposition"] == "review"
    assert review["subject_id"] == SUBJECT_IDS[0]
    assert review["opposing_unit_count"] == 1
    assert abstain["disposition"] == "abstain"
    assert abstain["subject_id"] is None


def test_execution_authority_allows_only_local_pilot_actions() -> None:
    preview = execution.preview_plan0056_execution(
        p0_authority={
            "preview_content_sha256": execution.P0_CONTENT_SHA256,
            "manifest_sha256": execution.P0_MANIFEST_SHA256,
            "source_count": 1,
            "source_set_sha256": "a" * 64,
            "allowlisted_subject_ids": list(SUBJECT_IDS),
            "profile_set_sha256": "b" * 64,
            "identity_state_before_sha256": "c" * 64,
            "idempotent_replay": True,
        },
        repository_authority={
            "commit": "d" * 40,
            "clean": True,
            "upstream_ahead": 0,
            "upstream_behind": 0,
        },
        local_runtime={
            "runtime_sha256": hashlib.sha256(b"local-runtime").hexdigest(),
            "network_required": False,
            "diarization_model_local": True,
            "transcription_model_local": True,
            "compute_device": "cuda",
            "compute_device_name": "test-gpu",
        },
        threshold_units=[
            {
                "candidate_id": candidate,
                "method_id": method,
                "threshold": 0.5,
                "temperature": 0.1,
            }
            for candidate in CANDIDATES
            for method in METHODS
        ],
    )

    assert preview["action_vector"]["run_local_diarization"] is True
    assert preview["action_vector"]["run_nine_acoustic_matrices"] is True
    assert preview["action_vector"]["prepare_human_review"] is True
    assert preview["action_vector"]["write_external_provider"] is False
    assert preview["action_vector"]["mutate_profiles_or_references"] is False
    assert preview["contains_pilot_outcome_gold"] is False


def test_execution_authority_freezes_and_replays_privately(tmp_path: Path) -> None:
    preview = execution.preview_plan0056_execution(
        p0_authority={
            "preview_content_sha256": execution.P0_CONTENT_SHA256,
            "manifest_sha256": execution.P0_MANIFEST_SHA256,
            "source_count": 1,
            "source_set_sha256": "a" * 64,
            "allowlisted_subject_ids": list(SUBJECT_IDS),
            "profile_set_sha256": "b" * 64,
            "identity_state_before_sha256": "c" * 64,
            "idempotent_replay": True,
        },
        repository_authority={
            "commit": "d" * 40,
            "clean": True,
            "upstream_ahead": 0,
            "upstream_behind": 0,
        },
        local_runtime={
            "runtime_sha256": hashlib.sha256(b"local-runtime").hexdigest(),
            "network_required": False,
            "diarization_model_local": True,
            "transcription_model_local": True,
            "compute_device": "cuda",
            "compute_device_name": "test-gpu",
        },
        threshold_units=[
            {
                "candidate_id": candidate,
                "method_id": method,
                "threshold": 0.5,
                "temperature": 0.1,
            }
            for candidate in CANDIDATES
            for method in METHODS
        ],
    )

    receipt = execution.freeze_plan0056_execution_authority(
        preview,
        expected_content_sha256=preview["content_sha256"],
        runtime_root=tmp_path / "runtime",
    )
    replay = execution.replay_plan0056_execution_authority(
        preview["content_sha256"], runtime_root=tmp_path / "runtime"
    )

    assert receipt["did_decode_audio"] is False
    assert replay["idempotent_replay"] is True
    assert ((tmp_path / "runtime").stat().st_mode & 0o777) == 0o700
