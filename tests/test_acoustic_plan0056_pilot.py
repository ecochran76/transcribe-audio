from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

import acoustic_plan0056_pilot as pilot


SUBJECT_IDS = (
    "subject-7c24e8f41409c6f517291fe7",
    "subject-df34bc192c07bd86566fff12",
)


def profile_inventory() -> tuple[list[dict], dict]:
    profiles = [
        {
            "profile_id": f"profile-{candidate}-{subject}",
            "person_ref_id": subject,
            "candidate_id": candidate,
            "generation_id": f"generation-{subject}",
            "generation_sha256": hashlib.sha256(subject.encode()).hexdigest(),
        }
        for candidate in (
            "speechbrain_ecapa_tdnn",
            "wespeaker_campplus",
            "wespeaker_resnet34",
        )
        for subject in SUBJECT_IDS
    ]
    return profiles, {
        "profile_count": 6,
        "subject_count": 2,
        "candidate_count": 3,
        "profile_set_sha256": hashlib.sha256(b"profiles").hexdigest(),
        "model_asset_set_sha256": hashlib.sha256(b"assets").hexdigest(),
    }


def test_proposals_accept_only_exact_allowlisted_subject_ids() -> None:
    accepted = pilot.validate_pilot_proposals(
        {
            "proposals": [
                {
                    "speaker_ref": "recording-1 / speaker-A",
                    "disposition": "assign",
                    "subject_id": SUBJECT_IDS[0],
                    "confidence_band": "medium",
                    "rationale": "The conservative acoustic rule passed.",
                }
            ]
        },
        expected_speaker_refs=("recording-1 / speaker-A",),
        allowlisted_subject_ids=SUBJECT_IDS,
    )

    assert accepted["proposals"][0]["subject_id"] == SUBJECT_IDS[0]
    assert accepted["contains_display_names"] is False

    for variant in (
        "Eric Cochran",
        "Eric W. Cochran",
        "eric-cochran",
        "gws:people/eric-cochran",
    ):
        with pytest.raises(pilot.Plan0056PilotError, match="allowlisted subject ID"):
            pilot.validate_pilot_proposals(
                {
                    "proposals": [
                        {
                            "speaker_ref": "recording-1 / speaker-A",
                            "disposition": "assign",
                            "subject_id": variant,
                            "confidence_band": "medium",
                            "rationale": "A name or provider variant must not become identity.",
                        }
                    ]
                },
                expected_speaker_refs=("recording-1 / speaker-A",),
                allowlisted_subject_ids=SUBJECT_IDS,
            )


def test_role_only_speakers_abstain_without_creating_an_identity() -> None:
    frozen = pilot.validate_pilot_proposals(
        {
            "proposals": [
                {
                    "speaker_ref": "recording-1 / meeting-host",
                    "disposition": "abstain",
                    "subject_id": None,
                    "confidence_band": "none",
                    "rationale": "The diarized voice does not match an enrolled subject.",
                }
            ]
        },
        expected_speaker_refs=("recording-1 / meeting-host",),
        allowlisted_subject_ids=SUBJECT_IDS,
    )

    assert frozen["proposals"][0]["subject_id"] is None
    assert frozen["will_apply_assignments"] is False

    with pytest.raises(pilot.Plan0056PilotError, match="must not carry"):
        pilot.validate_pilot_proposals(
            {
                "proposals": [
                    {
                        "speaker_ref": "recording-1 / meeting-host",
                        "disposition": "abstain",
                        "subject_id": "meeting-host",
                        "confidence_band": "none",
                        "rationale": "A role-only label cannot become an identity.",
                    }
                ]
            },
            expected_speaker_refs=("recording-1 / meeting-host",),
            allowlisted_subject_ids=SUBJECT_IDS,
        )


def test_authority_rejects_a_source_present_in_prior_evidence(tmp_path: Path) -> None:
    source = tmp_path / "pilot.m4a"
    source.write_bytes(b"fresh-pilot-audio")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    prior = tmp_path / "prior"
    prior.mkdir()
    (prior / "evidence.json").write_text(
        json.dumps({"source_sha256": digest}), encoding="utf-8"
    )

    with pytest.raises(pilot.Plan0056PilotError, match="prior evidence overlap"):
        pilot.preview_plan0056_authority(
            source_paths=(source,),
            prior_root=prior,
            profile_inventory=profile_inventory(),
            identity_state_snapshot={"snapshot_sha256": "a" * 64},
            repository_authority={
                "commit": "b" * 40,
                "clean": True,
                "upstream_ahead": 0,
                "upstream_behind": 0,
            },
            probe=lambda _path: {
                "duration_seconds": 120.0,
                "codec_name": "aac",
                "sample_rate": 48_000,
                "channels": 2,
            },
        )


def test_authority_freezes_two_subjects_and_all_negative_actions(tmp_path: Path) -> None:
    source = tmp_path / "pilot.m4a"
    source.write_bytes(b"fresh-pilot-audio")
    prior = tmp_path / "prior"
    prior.mkdir()
    (prior / "evidence.json").write_text(
        json.dumps({"source_sha256": "f" * 64}), encoding="utf-8"
    )

    preview = pilot.preview_plan0056_authority(
        source_paths=(source,),
        prior_root=prior,
        profile_inventory=profile_inventory(),
        identity_state_snapshot={"snapshot_sha256": "a" * 64},
        repository_authority={
            "commit": "b" * 40,
            "clean": True,
            "upstream_ahead": 0,
            "upstream_behind": 0,
        },
        probe=lambda _path: {
            "duration_seconds": 120.0,
            "codec_name": "aac",
            "sample_rate": 48_000,
            "channels": 2,
        },
    )

    assert preview["allowlisted_subject_ids"] == sorted(SUBJECT_IDS)
    assert preview["profile_summary"]["profile_count"] == 6
    assert preview["source_count"] == 1
    assert not any(preview["action_vector"].values())
    assert preview["contains_pilot_outcome_gold"] is False
    assert preview["scoring_policy"]["assignment_minimum_supporting_units"] == 6
    assert preview["scoring_policy"]["assignment_maximum_opposing_units"] == 0


def test_identity_state_snapshot_is_read_only_and_complete(tmp_path: Path) -> None:
    primary = tmp_path / "primary.sqlite3"
    knowledge = tmp_path / "knowledge.sqlite3"
    profiles = tmp_path / "profiles.sqlite3"
    references = tmp_path / "references.sqlite3"

    with sqlite3.connect(primary) as connection:
        for table in ("contacts", "speaker_assignments", "speaker_assignment_audits"):
            connection.execute(f"CREATE TABLE {table} (id TEXT PRIMARY KEY)")
        connection.execute("INSERT INTO contacts VALUES ('contact-1')")
    with sqlite3.connect(knowledge) as connection:
        for table in (
            "knowledge_people",
            "knowledge_external_identities",
            "knowledge_relationships",
            "knowledge_current_person_profiles",
            "knowledge_review_decisions",
        ):
            connection.execute(f"CREATE TABLE {table} (id TEXT PRIMARY KEY)")
    with sqlite3.connect(profiles) as connection:
        connection.execute(
            "CREATE TABLE profiles (profile_id TEXT, person_ref_id TEXT, "
            "generation_id TEXT, lifecycle_state TEXT)"
        )
        connection.execute(
            "INSERT INTO profiles VALUES ('p1', 's1', 'g1', 'active')"
        )
    with sqlite3.connect(references) as connection:
        for table in ("profiles", "generations", "person_heads", "source_claims", "descendants"):
            connection.execute(f"CREATE TABLE {table} (id TEXT PRIMARY KEY)")

    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in (primary, knowledge, profiles, references)}
    snapshot = pilot.snapshot_identity_state(
        primary_store=primary,
        knowledge_store=knowledge,
        profile_store=profiles,
        reference_store=references,
    )
    after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in before}

    assert snapshot["primary"]["contacts"] == 1
    assert snapshot["knowledge"]["knowledge_relationships"] == 0
    assert snapshot["acoustic_profiles"]["active_profiles"] == 1
    assert snapshot["acoustic_profiles"]["distinct_subjects"] == 1
    assert snapshot["references"]["generations"] == 0
    assert snapshot["snapshot_sha256"]
    assert after == before


def test_frozen_authority_replays_and_detects_source_drift(tmp_path: Path) -> None:
    source = tmp_path / "pilot.m4a"
    source.write_bytes(b"fresh-pilot-audio")
    prior = tmp_path / "prior"
    prior.mkdir()
    (prior / "evidence.json").write_text(
        json.dumps({"source_sha256": "f" * 64}), encoding="utf-8"
    )
    preview = pilot.preview_plan0056_authority(
        source_paths=(source,),
        prior_root=prior,
        profile_inventory=profile_inventory(),
        identity_state_snapshot={"snapshot_sha256": "a" * 64},
        repository_authority={
            "commit": "b" * 40,
            "clean": True,
            "upstream_ahead": 0,
            "upstream_behind": 0,
        },
        probe=lambda _path: {
            "duration_seconds": 120.0,
            "codec_name": "aac",
            "sample_rate": 48_000,
            "channels": 2,
        },
    )
    runtime = tmp_path / "runtime"

    receipt = pilot.freeze_plan0056_authority(
        preview,
        expected_content_sha256=preview["content_sha256"],
        runtime_root=runtime,
    )
    replay = pilot.replay_plan0056_authority(
        preview["content_sha256"], runtime_root=runtime
    )

    assert receipt["did_decode_audio"] is False
    assert replay["idempotent_replay"] is True
    assert (runtime.stat().st_mode & 0o777) == 0o700

    source.write_bytes(b"changed-pilot-audio")
    with pytest.raises(pilot.Plan0056PilotError, match="source drifted"):
        pilot.replay_plan0056_authority(
            preview["content_sha256"], runtime_root=runtime
        )


def test_portable_authority_omits_private_paths_and_profile_rows(tmp_path: Path) -> None:
    source = tmp_path / "pilot.m4a"
    source.write_bytes(b"fresh-pilot-audio")
    prior = tmp_path / "prior"
    prior.mkdir()
    (prior / "evidence.json").write_text(
        json.dumps({"source_sha256": "f" * 64}), encoding="utf-8"
    )
    preview = pilot.preview_plan0056_authority(
        source_paths=(source,),
        prior_root=prior,
        profile_inventory=profile_inventory(),
        identity_state_snapshot={"snapshot_sha256": "a" * 64},
        repository_authority={
            "commit": "b" * 40,
            "clean": True,
            "upstream_ahead": 0,
            "upstream_behind": 0,
        },
        probe=lambda _path: {
            "duration_seconds": 120.0,
            "codec_name": "aac",
            "sample_rate": 48_000,
            "channels": 2,
        },
    )

    portable = pilot.portable_authority(preview)
    serialized = json.dumps(portable)
    assert str(source) not in serialized
    assert "profile-" not in serialized
    assert portable["content_sha256"] == preview["content_sha256"]
    assert portable["source_count"] == 1
