from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
from pathlib import Path

import pytest

import speaker_identity_plan0064_p0 as p0


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write(path: Path, value: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    path.write_bytes(value)
    os.chmod(path, 0o600)
    return _hash_bytes(value)


def _json_hash(value: object) -> str:
    return p0._canonical_hash(value)


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def _reference_store(root: Path) -> None:
    with _connect(root / "references.sqlite3") as connection:
        connection.executescript(
            """
            CREATE TABLE profiles (
                profile_id TEXT PRIMARY KEY, person_ref_id TEXT, status TEXT,
                head_generation_id TEXT, head_version INTEGER,
                event_sequence INTEGER, descendant_count INTEGER,
                last_event_sha256 TEXT, created_at TEXT, deleted_at TEXT
            );
            CREATE TABLE generations (
                generation_id TEXT PRIMARY KEY, profile_id TEXT, sequence INTEGER,
                predecessor_generation_id TEXT, status TEXT,
                eligible_for_materialization INTEGER, manifest_json TEXT,
                manifest_sha256 TEXT, created_at TEXT
            );
            CREATE TABLE person_heads (
                person_ref_id TEXT PRIMARY KEY, profile_id TEXT,
                generation_id TEXT, status TEXT, version INTEGER
            );
            CREATE TABLE events (
                event_id TEXT PRIMARY KEY, profile_id TEXT, sequence INTEGER,
                action TEXT, generation_id TEXT, previous_event_sha256 TEXT,
                payload_json TEXT, event_sha256 TEXT, created_at TEXT
            );
            CREATE TABLE source_claims (
                source_key TEXT PRIMARY KEY, source_sha256 TEXT,
                start_seconds REAL, end_seconds REAL, person_ref_id TEXT,
                first_profile_id TEXT, first_generation_id TEXT
            );
            CREATE TABLE descendants (
                descendant_id TEXT PRIMARY KEY, profile_id TEXT,
                generation_id TEXT, generation_sha256 TEXT,
                artifact_sha256 TEXT, registered_at TEXT, state TEXT
            );
            """
        )
        for index, subject in enumerate(("person-a", "voice-b", "voice-c"), start=1):
            profile_id = f"reference-{index}"
            generation_id = f"generation-{index}"
            source = {
                "reference_id": f"source-{index}",
                "source_key": f"segment-{index:064x}"[-64:],
                "source_sha256": _hash_bytes(f"development-{index}".encode()),
                "recording_id": f"development-recording-{index}",
                "conversation_id": f"development-conversation-{index}",
                "speaker_label_id": "SPEAKER_1",
                "session_id": f"session-{index}",
                "start_seconds": 0.0,
                "end_seconds": 10.0,
            }
            manifest = {
                "generation_id": generation_id,
                "person_ref_id": subject,
                "eligible_for_materialization": True,
                "source_set_sha256": _json_hash([source]),
                "sources": [source],
            }
            manifest_sha = _json_hash(manifest)
            connection.execute(
                "INSERT INTO profiles VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    profile_id,
                    subject,
                    "active",
                    generation_id,
                    1,
                    1,
                    3,
                    f"event-{index}",
                    "2026-08-09T00:00:00Z",
                    None,
                ),
            )
            connection.execute(
                "INSERT INTO generations VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    generation_id,
                    profile_id,
                    1,
                    None,
                    "active",
                    1,
                    json.dumps(manifest, sort_keys=True),
                    manifest_sha,
                    "2026-08-09T00:00:00Z",
                ),
            )
            connection.execute(
                "INSERT INTO person_heads VALUES (?,?,?,?,?)",
                (subject, profile_id, generation_id, "active", 1),
            )
            connection.execute(
                "INSERT INTO source_claims VALUES (?,?,?,?,?,?,?)",
                (
                    source["source_key"],
                    source["source_sha256"],
                    0.0,
                    10.0,
                    subject,
                    profile_id,
                    generation_id,
                ),
            )
            for model_index, candidate_id in enumerate(("m1", "m2", "m3"), start=1):
                connection.execute(
                    "INSERT INTO descendants VALUES (?,?,?,?,?,?,?)",
                    (
                        f"descendant-{index}-{model_index}",
                        profile_id,
                        generation_id,
                        manifest_sha,
                        f"{index * 10 + model_index:064x}",
                        "2026-08-09T00:00:00Z",
                        "eligible",
                    ),
                )


def _profile_store(root: Path, reference_root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    os.chmod(root, 0o700)
    with sqlite3.connect(reference_root / "references.sqlite3") as reference_db:
        generations = {
            row[0]: row[1]
            for row in reference_db.execute(
                "SELECT generation_id, manifest_sha256 FROM generations"
            )
        }
    with _connect(root / "profiles.sqlite3") as connection:
        connection.execute(
            """
            CREATE TABLE profiles (
                profile_id TEXT PRIMARY KEY, descendant_id TEXT UNIQUE,
                person_ref_id TEXT, p3_profile_id TEXT, generation_id TEXT,
                generation_sha256 TEXT, candidate_id TEXT, model_revision TEXT,
                preprocessing_json TEXT, artifact_path TEXT, artifact_sha256 TEXT,
                vector_dimension INTEGER, window_count INTEGER,
                session_count INTEGER, dispersion REAL, lifecycle_state TEXT,
                created_at TEXT, updated_at TEXT,
                invalidation_receipt_sha256 TEXT, tombstone_path TEXT,
                replacement_profile_id TEXT, state_receipt_sha256 TEXT,
                profile_manifest_path TEXT, profile_manifest_sha256 TEXT
            )
            """
        )
        for index, subject in enumerate(("person-a", "voice-b", "voice-c"), start=1):
            for model_index, candidate_id in enumerate(("m1", "m2", "m3"), start=1):
                profile_id = f"model-profile-{index}-{model_index}"
                artifact_path = root / "profiles" / f"{profile_id}.bin"
                manifest_path = root / "profiles" / f"{profile_id}.json"
                artifact_sha = _write(artifact_path, bytes(range(16)))
                profile_manifest = {
                    "profile_id": profile_id,
                    "descendant_id": f"descendant-{index}-{model_index}",
                    "person_ref_id": subject,
                    "p3_profile_id": f"reference-{index}",
                    "generation_id": f"generation-{index}",
                    "generation_sha256": generations[f"generation-{index}"],
                    "candidate_id": candidate_id,
                    "model_revision": f"revision-{candidate_id}",
                    "preprocessing": {},
                    "artifact_path": str(artifact_path),
                    "artifact_sha256": artifact_sha,
                    "vector_dimension": 4,
                    "window_count": 1,
                    "session_count": 1,
                    "dispersion": 0.1,
                    "contains_raw_biometric_values": False,
                }
                manifest_sha = _write(
                    manifest_path,
                    json.dumps(profile_manifest, sort_keys=True).encode(),
                )
                lifecycle = {
                    "profile_id": profile_id,
                    "descendant_id": f"descendant-{index}-{model_index}",
                    "artifact_sha256": artifact_sha,
                    "profile_manifest_sha256": manifest_sha,
                    "to_state": "active",
                    "replacement_profile_id": None,
                    "will_perform_external_write": False,
                }
                lifecycle_sha = _json_hash(lifecycle)
                _write(
                    root / "authority" / f"{lifecycle_sha}.json",
                    json.dumps(lifecycle, sort_keys=True).encode(),
                )
                with sqlite3.connect(
                    reference_root / "references.sqlite3"
                ) as reference_connection:
                    reference_connection.execute(
                        "UPDATE descendants SET artifact_sha256=? WHERE descendant_id=?",
                        (artifact_sha, f"descendant-{index}-{model_index}"),
                    )
                connection.execute(
                    "INSERT INTO profiles VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        profile_id,
                        f"descendant-{index}-{model_index}",
                        subject,
                        f"reference-{index}",
                        f"generation-{index}",
                        generations[f"generation-{index}"],
                        candidate_id,
                        f"revision-{candidate_id}",
                        "{}",
                        str(artifact_path),
                        artifact_sha,
                        4,
                        1,
                        1,
                        0.1,
                        "active",
                        "2026-08-09T00:00:00Z",
                        "2026-08-09T00:00:00Z",
                        None,
                        None,
                        None,
                        lifecycle_sha,
                        str(manifest_path),
                        manifest_sha,
                    ),
                )


def _knowledge_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE knowledge_people (
            id TEXT PRIMARY KEY, status TEXT, primary_name TEXT
        );
        CREATE TABLE knowledge_source_records (
            id TEXT PRIMARY KEY, person_id TEXT, label TEXT,
            provider_kind TEXT, content_hash TEXT
        );
        CREATE TABLE knowledge_observations (
            id TEXT PRIMARY KEY, observation_type TEXT, subject_type TEXT,
            subject_id TEXT, source_type TEXT, source_id TEXT,
            conversation_id TEXT, source_event_at TEXT, observed_at TEXT,
            retrieved_at TEXT, valid_from TEXT, valid_to TEXT,
            review_state TEXT, payload_json TEXT, content_hash TEXT,
            created_at TEXT
        );
        CREATE TABLE knowledge_current_person_profiles (
            person_id TEXT PRIMARY KEY, input_watermark TEXT
        );
        """
    )
    for person_id in ("person-a", "person-b"):
        connection.execute(
            "INSERT INTO knowledge_people VALUES (?,?,?)",
            (person_id, "resolved", person_id),
        )
        connection.execute(
            "INSERT INTO knowledge_source_records VALUES (?,?,?,?,?)",
            (f"source-{person_id}", person_id, person_id, "test", f"hash-{person_id}"),
        )
        connection.execute(
            "INSERT INTO knowledge_current_person_profiles VALUES (?,?)",
            (person_id, f"watermark-{person_id}"),
        )
    payload = {
        "acoustic_subject_id": "voice-b",
        "person_id": "person-b",
        "active_binding": True,
    }
    connection.execute(
        "INSERT INTO knowledge_observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            "binding-1",
            "reviewed_voice_subject_binding_confirmed",
            "person",
            "person-b",
            "human_review",
            "review-1",
            "",
            "2026-08-09T00:00:00Z",
            "2026-08-09T00:00:00Z",
            "",
            "",
            "",
            "accepted",
            json.dumps(payload, sort_keys=True),
            _json_hash(payload),
            "2026-08-09T00:00:00Z",
        ),
    )


def _transcript_store(root: Path, cases: list[dict[str, object]]) -> None:
    with _connect(root / "transcripts.sqlite3") as connection:
        connection.executescript(
            """
            CREATE TABLE documents (
                id TEXT PRIMARY KEY, kind TEXT, title TEXT, source_path TEXT,
                stored_path TEXT, artifact_sha256 TEXT, generated_at TEXT,
                text_content TEXT, json_payload TEXT, metadata_json TEXT,
                embedding_json TEXT, created_at TEXT, updated_at TEXT
            );
            CREATE TABLE blobs (
                id TEXT PRIMARY KEY, role TEXT, original_path TEXT,
                stored_path TEXT, sha256 TEXT, mime_type TEXT,
                bytes INTEGER, created_at TEXT, updated_at TEXT
            );
            CREATE TABLE document_blobs (
                document_id TEXT, blob_id TEXT, role TEXT, created_at TEXT,
                PRIMARY KEY(document_id, blob_id, role)
            );
            """
        )
        _knowledge_schema(connection)
        for index, case in enumerate(cases, start=1):
            document_id = str(case.get("document_id") or f"document-{index}")
            transcript_path = root / "artifacts" / f"{document_id}.transcript.json"
            utterances = case.get(
                "utterances",
                [
                    {"speaker": "SPEAKER_1", "text": "hello"},
                    {"speaker": "SPEAKER_2", "text": "world"},
                ],
            )
            payload = {
                "recording_start": f"2020-01-{index:02d}T00:00:00Z",
                "recording_id": f"recording-{index}",
                "conversation_id": f"conversation-{index}",
                "utterances": utterances,
                "event": {"summary": "private"} if case.get("event", True) else None,
            }
            transcript_bytes = json.dumps(payload, sort_keys=True).encode()
            transcript_sha = _write(transcript_path, transcript_bytes)
            connection.execute(
                "INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    document_id,
                    "transcript",
                    document_id,
                    str(transcript_path),
                    str(transcript_path),
                    transcript_sha,
                    payload["recording_start"],
                    "",
                    json.dumps(payload, sort_keys=True),
                    "{}",
                    "[]",
                    payload["recording_start"],
                    payload["recording_start"],
                ),
            )
            if case.get("media", True):
                media_path = root / "blobs" / f"{document_id}.wav"
                media_bytes = bytes(case.get("media_bytes") or f"audio-{index}".encode())
                media_sha = _write(media_path, media_bytes)
                blob_id = f"blob-{index}"
                connection.execute(
                    "INSERT INTO blobs VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        blob_id,
                        "source_recording",
                        str(media_path),
                        str(media_path),
                        media_sha,
                        "audio/wav",
                        len(media_bytes),
                        payload["recording_start"],
                        payload["recording_start"],
                    ),
                )
                connection.execute(
                    "INSERT INTO document_blobs VALUES (?,?,?,?)",
                    (
                        document_id,
                        blob_id,
                        "source_recording",
                        payload["recording_start"],
                    ),
                )


def _fixture(tmp_path: Path, cases: list[dict[str, object]]) -> dict[str, Path]:
    roots = {
        "reference_root": tmp_path / "references",
        "profile_root": tmp_path / "profiles",
        "transcript_root": tmp_path / "transcripts",
        "prior_campaign_root": tmp_path / "campaigns",
        "runtime_root": tmp_path / "runtime",
    }
    _reference_store(roots["reference_root"])
    _profile_store(roots["profile_root"], roots["reference_root"])
    _transcript_store(roots["transcript_root"], cases)
    return roots


def _repository() -> dict[str, object]:
    return {
        "head": "a" * 40,
        "module_commit": "a" * 40,
        "module_name": p0.MODULE_NAME,
        "module_sha256": "b" * 64,
        "module_blob_matches": True,
        "clean": True,
        "upstream_behind": 0,
        "upstream_ahead": 0,
    }


def _input_roots(roots: dict[str, Path]) -> dict[str, Path]:
    return {
        key: roots[key]
        for key in (
            "transcript_root",
            "reference_root",
            "profile_root",
            "prior_campaign_root",
        )
    }


def test_manifest_uses_complete_inventory_and_abstains_unbound(tmp_path: Path) -> None:
    roots = _fixture(tmp_path, [{}, {}, {}])

    manifest = p0.build_p0_manifest(
        **_input_roots(roots),
        evaluation_limit=2,
        repository=_repository(),
    )

    summary = p0._public_summary(manifest)
    assert summary["active_reference_count"] == 3
    assert summary["active_profile_count"] == 9
    assert summary["identity_ready_subject_count"] == 2
    assert summary["unbound_subject_count"] == 1
    assert summary["identity_ready_profile_count"] == 6
    assert summary["unbound_active_profile_count"] == 3
    assert summary["selected_recording_count"] == 2
    assert not any(summary["action_counts"].values())
    unbound = [
        item
        for item in manifest["profile_inventory"]["active_profiles"]
        if item["person_ref_id"] == "voice-c"
    ]
    assert len(unbound) == 3
    assert all(item["identity_candidate_eligible"] is False for item in unbound)


def test_withdrawn_profiles_stay_counted_but_leave_active_matrix(tmp_path: Path) -> None:
    roots = _fixture(tmp_path, [{}, {}])
    with sqlite3.connect(roots["profile_root"] / "profiles.sqlite3") as connection:
        connection.execute(
            "UPDATE profiles SET lifecycle_state='withdrawn' WHERE person_ref_id='voice-c'"
        )

    manifest = p0.build_p0_manifest(
        **_input_roots(roots),
        evaluation_limit=1,
        repository=_repository(),
    )

    counts = manifest["profile_inventory"]["profile_count_by_state"]
    assert counts == {"active": 6, "withdrawn": 3}
    assert manifest["profile_inventory"]["subject_count"] == 2


def test_profile_reference_drift_fails_closed(tmp_path: Path) -> None:
    roots = _fixture(tmp_path, [{}, {}])
    with sqlite3.connect(roots["profile_root"] / "profiles.sqlite3") as connection:
        connection.execute(
            "UPDATE profiles SET descendant_id='missing' WHERE profile_id='model-profile-1-1'"
        )

    with pytest.raises(p0.Plan0064P0Error) as raised:
        p0.build_p0_manifest(
            **_input_roots(roots),
            evaluation_limit=1,
            repository=_repository(),
        )
    assert raised.value.reason_code == "profile_reference_binding_drift"


def test_cohort_reason_codes_overlap_exposure_repeat_and_missing_media(
    tmp_path: Path,
) -> None:
    shared = b"same-recording"
    cases = [
        {"document_id": "prior"},
        {"document_id": "overlap", "media_bytes": b"development-1"},
        {"document_id": "duplicate-a", "media_bytes": shared},
        {"document_id": "duplicate-b", "media_bytes": shared},
        {"document_id": "missing-media", "media": False},
        {"document_id": "selected-a"},
        {"document_id": "selected-b", "event": False},
    ]
    roots = _fixture(tmp_path, cases)
    (roots["prior_campaign_root"] / "campaign-a" / "gold" / "prior").mkdir(
        parents=True
    )
    manifest = p0.build_p0_manifest(
        **_input_roots(roots),
        evaluation_limit=2,
        repository=_repository(),
    )

    cohort = manifest["evaluation_cohort"]
    assert cohort["selected_document_ids"] == ["duplicate-a", "selected-a"]
    counts = cohort["reason_code_counts"]
    assert counts["prior_identity_evidence_exposure"] == 1
    assert counts["development_recording_overlap"] == 1
    assert counts["repeated_recording_hash"] == 1
    assert counts["source_media_unavailable"] == 1


def test_incomplete_candidate_denominator_fails_closed(tmp_path: Path) -> None:
    roots = _fixture(
        tmp_path,
        [{"utterances": []}, {"media": False}, {"utterances": []}],
    )
    with pytest.raises(p0.Plan0064P0Error) as raised:
        p0.build_p0_manifest(
            **_input_roots(roots),
            evaluation_limit=2,
            repository=_repository(),
        )
    assert raised.value.reason_code == "incomplete_candidate_denominator"


def test_freeze_and_replay_are_private_idempotent_and_zero_effect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = _fixture(tmp_path, [{}, {}])
    monkeypatch.setattr(p0, "repository_authority", lambda **_: _repository())
    common = {
        key: roots[key]
        for key in (
            "transcript_root",
            "reference_root",
            "profile_root",
            "prior_campaign_root",
        )
    }
    manifest = p0.build_p0_manifest(
        **common, evaluation_limit=1, repository=_repository()
    )

    frozen = p0.freeze_p0(
        expected_content_sha256=manifest["content_sha256"],
        runtime_root=roots["runtime_root"],
        evaluation_limit=1,
        **common,
    )
    replayed = p0.replay_p0(
        content_sha256=manifest["content_sha256"],
        runtime_root=roots["runtime_root"],
        evaluation_limit=1,
        **common,
    )

    assert frozen["status"] == "p0_frozen_zero_effect"
    assert frozen["idempotent_replay"] is False
    assert replayed["idempotent_replay"] is True
    for key in ("private_manifest_path", "private_receipt_path"):
        assert stat.S_IMODE(Path(frozen[key]).stat().st_mode) == 0o600
    assert not any(replayed["summary"]["action_counts"].values())
