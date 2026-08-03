import json
from pathlib import Path

import pytest

import acoustic_generation4_cohort as cohort


AUTHORITY = {
    "g0_preview_sha256": "a" * 64,
    "g0_manifest_sha256": "b" * 64,
    "media_preview_sha256": "c" * 64,
    "qualified_set_sha256": "d" * 64,
}
REPOSITORY = {
    "commit": "1" * 40,
    "module_name": cohort.MODULE_NAME,
    "module_sha256": "2" * 64,
    "clean": True,
    "upstream_ahead": 0,
    "upstream_behind": 0,
}


def _audio(root: Path, index: int) -> tuple[Path, dict]:
    path = root / f"audio-{index}.m4a"
    path.write_bytes(f"audio-{index}".encode())
    return path, {"path": str(path), "source_sha256": cohort.sha256_file(path)}


def _transcript(root: Path, audio: Path, index: int, labels=("A",)) -> Path:
    path = root / f"transcript-{index}.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_media_path": str(audio),
                "conversation_id": f"conversation-{index}",
                "recording_id": f"recording-{index}",
                "utterances": [
                    {"speaker": label, "text": "private"} for label in labels
                ],
                "event": {},
            }
        )
    )
    return path


def _bind_source_authority(monkeypatch, qualified: list[dict]) -> None:
    membership = [
        {**item, "authority_origin": item.get("authority_origin", "original")}
        for item in qualified
    ]
    monkeypatch.setattr(
        cohort,
        "_source_authority",
        lambda: (dict(AUTHORITY), membership),
    )


def _media_manifest(root: Path, count: int) -> tuple[Path, str, str]:
    root.mkdir(mode=0o700)
    results = [
        {
            "path": str(root / f"audio-{index}.m4a"),
            "source_sha256": f"{index + 1:064x}",
            "status": "qualified",
        }
        for index in range(count)
    ]
    preview_sha256 = "e" * 64
    manifest = {
        "schema_version": cohort.media.MANIFEST_SCHEMA,
        "status": "frozen",
        "preview": {
            "content_sha256": preview_sha256,
            "candidate_count": count,
            "qualified_count": count,
            "private_results": results,
        },
    }
    path = root / "private-manifest.json"
    path.write_text(json.dumps(manifest, sort_keys=True))
    path.chmod(0o600)
    qualified_hash = cohort._canonical_hash(
        sorted(item["source_sha256"] for item in results)
    )
    return path, preview_sha256, qualified_hash


def test_manifest_membership_binds_each_member_to_origin(tmp_path: Path) -> None:
    root = tmp_path / "private"
    path, preview_hash, qualified_hash = _media_manifest(root, 2)

    members = cohort._manifest_membership(
        path,
        private_root=root,
        expected_manifest_sha256=cohort.sha256_file(path),
        expected_preview_sha256=preview_hash,
        expected_qualified_set_sha256=qualified_hash,
        expected_qualified_count=2,
        expected_candidate_count=2,
        authority_origin="supplemental",
    )

    assert [item["authority_origin"] for item in members] == [
        "supplemental",
        "supplemental",
    ]


def test_manifest_membership_rejects_manifest_tamper(tmp_path: Path) -> None:
    root = tmp_path / "private"
    path, preview_hash, qualified_hash = _media_manifest(root, 2)
    expected_manifest_hash = cohort.sha256_file(path)
    value = json.loads(path.read_text())
    value["preview"]["private_results"][0]["source_sha256"] = "f" * 64
    path.write_text(json.dumps(value, sort_keys=True))

    with pytest.raises(cohort.Generation4CohortError, match="manifest drifted"):
        cohort._manifest_membership(
            path,
            private_root=root,
            expected_manifest_sha256=expected_manifest_hash,
            expected_preview_sha256=preview_hash,
            expected_qualified_set_sha256=qualified_hash,
            expected_qualified_count=2,
            expected_candidate_count=2,
            authority_origin="supplemental",
        )


def test_manifest_membership_rejects_qualified_set_tamper(tmp_path: Path) -> None:
    root = tmp_path / "private"
    path, preview_hash, _ = _media_manifest(root, 2)

    with pytest.raises(cohort.Generation4CohortError, match="set hash drifted"):
        cohort._manifest_membership(
            path,
            private_root=root,
            expected_manifest_sha256=cohort.sha256_file(path),
            expected_preview_sha256=preview_hash,
            expected_qualified_set_sha256="f" * 64,
            expected_qualified_count=2,
            expected_candidate_count=2,
            authority_origin="supplemental",
        )


def test_manifest_membership_rejects_supplemental_pool_above_bound(
    tmp_path: Path,
) -> None:
    root = tmp_path / "private"
    path, preview_hash, qualified_hash = _media_manifest(root, 13)

    with pytest.raises(cohort.Generation4CohortError, match="manifest drifted"):
        cohort._manifest_membership(
            path,
            private_root=root,
            expected_manifest_sha256=cohort.sha256_file(path),
            expected_preview_sha256=preview_hash,
            expected_qualified_set_sha256=qualified_hash,
            expected_qualified_count=13,
            expected_candidate_count=13,
            authority_origin="supplemental",
        )


def test_pool_with_fewer_than_seven_linked_recordings_requests_supplement(
    tmp_path: Path, monkeypatch,
) -> None:
    tmp_path.chmod(0o700)
    qualified = []
    for index in range(10):
        audio, item = _audio(tmp_path, index)
        qualified.append(item)
        if index == 0:
            _transcript(tmp_path, audio, index)
    for index in range(10, 22):
        audio, _ = _audio(tmp_path, index)
        _transcript(tmp_path, audio, index, ("A", "B"))

    _bind_source_authority(monkeypatch, qualified)
    preview = cohort.preview_generation4_cohort(
        source_root=tmp_path,
        repository_authority=REPOSITORY,
    )

    assert preview["status"] == "supplemental_pool_requested"
    assert preview["qualified_with_exact_transcript_count"] == 1
    assert preview["qualified_without_exact_transcript_count"] == 9
    assert preview["supplemental_candidate_count"] == 12
    assert preview["action_vector"]["request_one_supplemental_pool"] is True
    assert preview["action_vector"]["freeze_g2_cohort"] is False
    portable = cohort._portable(preview)
    assert portable["contains_paths"] is False
    assert portable["contains_private_membership"] is False
    assert "private_evidence" not in portable


def test_absent_gold_does_not_consume_supplement_for_nine_linked_recordings(
    tmp_path: Path, monkeypatch,
) -> None:
    tmp_path.chmod(0o700)
    qualified = []
    for index in range(10):
        audio, item = _audio(tmp_path, index)
        qualified.append(item)
        if index < 9:
            _transcript(tmp_path, audio, index, ("A", "B"))

    _bind_source_authority(monkeypatch, qualified)
    preview = cohort.preview_generation4_cohort(
        source_root=tmp_path,
        repository_authority=REPOSITORY,
    )

    assert preview["status"] == "private_gold_review_required"
    assert preview["qualified_with_exact_transcript_count"] == 9
    assert preview["proposed_original_cohort_count"] == 7
    assert preview["supplemental_candidate_count"] == 0
    assert preview["action_vector"]["request_one_supplemental_pool"] is False
    assert preview["action_vector"]["complete_private_gold_review"] is True
    assert preview["population_feasibility"][
        "missing_gold_is_not_population_infeasibility"
    ] is True
    assert preview["population_feasibility"]["identity_session_coverage_status"] == (
        "unknown_pending_private_gold_review"
    )
    assert preview["delegation_receipt"] == {
        "status": "spawned",
        "lane": "G1A",
        "runtime_handle": "/root/g1a_cohort_feasibility",
        "terminal_status": "gold_review_gate",
        "returned_evidence_sha256": preview["delegation_receipt"][
            "returned_evidence_sha256"
        ],
        "primary_reconciliation": "pending_gold_review",
    }
    assert cohort.SHA256_RE.fullmatch(
        preview["delegation_receipt"]["returned_evidence_sha256"]
    )


def test_nine_linked_of_ten_passes_with_valid_seven_case_gold_subset(
    tmp_path: Path, monkeypatch,
) -> None:
    tmp_path.chmod(0o700)
    qualified = []
    transcripts = []
    for index in range(10):
        audio, item = _audio(tmp_path, index)
        qualified.append(item)
        if index < 9:
            transcripts.append(_transcript(tmp_path, audio, index, ("A", "B")))
    assignments = [
        ("p1", "enrolled-1"),
        ("p1", "enrolled-1"),
        ("p1", "enrolled-1"),
        ("p2", "enrolled-2"),
        ("p2", "enrolled-2"),
        ("p3", ""),
        ("p4", ""),
    ]
    cases = []
    for index, (person, enrolled) in enumerate(assignments):
        cases.append(
            {
                "source_sha256": qualified[index]["source_sha256"],
                "transcript_sha256": cohort.sha256_file(transcripts[index]),
                "conversation_id": f"conversation-{index}",
                "recording_id": f"recording-{index}",
                "speaker_gold": [
                    {
                        "speaker_label": "A",
                        "person_id": person,
                        "enrolled_subject_id": enrolled,
                    },
                    {"speaker_label": "B", "person_id": "p5"},
                ],
            }
        )
    gold_path = tmp_path / "private-gold.json"
    gold_path.write_text(
        json.dumps({"schema_version": cohort.GOLD_SCHEMA, "cases": cases})
    )
    gold_path.chmod(0o600)

    _bind_source_authority(monkeypatch, qualified)
    preview = cohort.preview_generation4_cohort(
        source_root=tmp_path,
        gold_path=gold_path,
        repository_authority=REPOSITORY,
    )

    assert preview["qualified_with_exact_transcript_count"] == 9
    assert preview["status"] == "passing_population_proposal"
    assert preview["proposed_original_cohort_count"] == 7
    assert preview["population"]["passing"] is True
    assert preview["population_feasibility"]["identity_session_coverage_status"] == (
        "proven_by_private_gold_subset"
    )
    assert preview["supplemental_candidate_count"] == 0
    assert preview["action_vector"]["request_one_supplemental_pool"] is False


def test_population_gate_accepts_seven_disjoint_cases_with_required_coverage() -> None:
    people = ["p1", "p2", "p3", "p4", "p5"]
    assignments = [
        ("p1", "enrolled-1"),
        ("p1", "enrolled-1"),
        ("p1", "enrolled-1"),
        ("p2", "enrolled-2"),
        ("p2", "enrolled-2"),
        ("p3", ""),
        ("p4", ""),
    ]
    cases = []
    expected = set()
    for index, (person, enrolled) in enumerate(assignments):
        source = f"source-{index}"
        expected.add(source)
        cases.append(
            {
                "source_sha256": source,
                "transcript_sha256": f"transcript-{index}",
                "conversation_id": f"conversation-{index}",
                "recording_id": f"recording-{index}",
                "speaker_gold": [
                    {
                        "speaker_label": "A",
                        "person_id": person,
                        "enrolled_subject_id": enrolled,
                    },
                    {
                        "speaker_label": "B",
                        "person_id": people[4],
                        "enrolled_subject_id": "",
                    },
                ],
            }
        )

    result = cohort.evaluate_population(cases, expected_sources=expected)

    assert result["passing"] is True
    assert result["conversation_count"] == 7
    assert result["person_count"] == 5
    assert result["enrolled_people_with_two_sessions_count"] == 2
    assert result["same_person_session_pair_count"] >= 4


def test_population_gate_fails_closed_on_duplicate_conversation() -> None:
    cases = [
        {
            "source_sha256": f"source-{index}",
            "transcript_sha256": f"transcript-{index}",
            "conversation_id": "same-conversation",
            "recording_id": f"recording-{index}",
            "speaker_gold": [{"speaker_label": "A", "person_id": "p1"}],
        }
        for index in range(2)
    ]
    result = cohort.evaluate_population(
        cases, expected_sources={"source-0", "source-1"}
    )
    assert result["passing"] is False
    assert result["overlap_count"] == 1
    assert result["gates"]["zero_overlap"] is False


def test_preview_rejects_gold_whose_labels_do_not_match_transcript(
    tmp_path: Path, monkeypatch
) -> None:
    tmp_path.chmod(0o700)
    audio, qualified = _audio(tmp_path, 0)
    transcript = _transcript(tmp_path, audio, 0, ("A", "B"))
    gold_path = tmp_path / "private-gold.json"
    gold_path.write_text(
        json.dumps(
            {
                "schema_version": cohort.GOLD_SCHEMA,
                "cases": [
                    {
                        "source_sha256": qualified["source_sha256"],
                        "transcript_sha256": cohort.sha256_file(transcript),
                        "conversation_id": "conversation-0",
                        "recording_id": "recording-0",
                        "speaker_gold": [
                            {"speaker_label": "A", "person_id": "person-1"}
                        ],
                    }
                ],
            }
        )
    )
    gold_path.chmod(0o600)

    _bind_source_authority(monkeypatch, [qualified])
    preview = cohort.preview_generation4_cohort(
        source_root=tmp_path,
        gold_path=gold_path,
        repository_authority=REPOSITORY,
    )

    assert preview["population"]["passing"] is False
    assert preview["population"]["overlap_count"] == 1


def test_preview_rejects_invalid_repository_binding(tmp_path: Path, monkeypatch) -> None:
    tmp_path.chmod(0o700)
    audio, qualified = _audio(tmp_path, 0)
    _transcript(tmp_path, audio, 0)
    _bind_source_authority(monkeypatch, [qualified])
    invalid = {**REPOSITORY, "clean": False}

    with pytest.raises(cohort.Generation4CohortError, match="repository authority"):
        cohort.preview_generation4_cohort(
            source_root=tmp_path,
            repository_authority=invalid,
        )


def test_apply_rejects_stale_repository_binding(tmp_path: Path, monkeypatch) -> None:
    tmp_path.chmod(0o700)
    audio, qualified = _audio(tmp_path, 0)
    _transcript(tmp_path, audio, 0)
    _bind_source_authority(monkeypatch, [qualified])
    preview = cohort.preview_generation4_cohort(
        source_root=tmp_path,
        repository_authority=REPOSITORY,
    )
    drifted = {**REPOSITORY, "module_sha256": "3" * 64}

    with pytest.raises(cohort.Generation4CohortError, match="stale"):
        cohort.apply_generation4_cohort(
            preview,
            expected_content_sha256=preview["content_sha256"],
            runtime_root=tmp_path / "runtime",
            source_root=tmp_path,
            repository_authority=drifted,
        )


def test_apply_replay_is_private_idempotent_and_detects_drift(
    tmp_path: Path, monkeypatch
) -> None:
    tmp_path.chmod(0o700)
    source = tmp_path / "source"
    source.mkdir(mode=0o700)
    qualified = []
    for index in range(2):
        audio, item = _audio(source, index)
        qualified.append(item)
        _transcript(source, audio, index)
    runtime = tmp_path / "runtime"
    _bind_source_authority(monkeypatch, qualified)
    preview = cohort.preview_generation4_cohort(
        source_root=source,
        repository_authority=REPOSITORY,
    )

    applied = cohort.apply_generation4_cohort(
        preview,
        expected_content_sha256=preview["content_sha256"],
        runtime_root=runtime,
        source_root=source,
        repository_authority=REPOSITORY,
    )
    replayed = cohort.replay_generation4_cohort(
        preview["content_sha256"],
        runtime_root=runtime,
        source_root=source,
        repository_authority=REPOSITORY,
    )

    paths = cohort._paths(runtime, preview["content_sha256"])
    assert applied["idempotent_replay"] is False
    assert replayed["idempotent_replay"] is True
    assert paths["run"].stat().st_mode & 0o777 == 0o700
    assert paths["manifest"].stat().st_mode & 0o777 == 0o600
    assert paths["receipt"].stat().st_mode & 0o777 == 0o600
    stored_receipt = json.loads(paths["receipt"].read_text())
    assert "private_evidence" not in stored_receipt
    assert {key: applied[key] for key in stored_receipt} == stored_receipt
    assert {key: replayed[key] for key in stored_receipt} == stored_receipt

    broadened = [
        *qualified,
        {
            "path": str(source / "unauthorized.m4a"),
            "source_sha256": "f" * 64,
        },
    ]
    _bind_source_authority(monkeypatch, broadened)
    with pytest.raises(cohort.Generation4CohortError, match="drifted"):
        cohort.replay_generation4_cohort(
            preview["content_sha256"],
            runtime_root=runtime,
            source_root=source,
            repository_authority=REPOSITORY,
        )

    _bind_source_authority(monkeypatch, qualified)
    source.joinpath("transcript-0.json").write_text("{}")
    with pytest.raises(cohort.Generation4CohortError, match="drifted"):
        cohort.replay_generation4_cohort(
            preview["content_sha256"],
            runtime_root=runtime,
            source_root=source,
            repository_authority=REPOSITORY,
        )
