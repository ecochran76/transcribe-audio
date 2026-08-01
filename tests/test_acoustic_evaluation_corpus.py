from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path

import pytest

import transcript_store
import acoustic_evaluation_corpus as corpus
from acoustic_evaluation_corpus import (
    FREEZE_TOKEN,
    HARDEN_TOKEN,
    CorpusError,
    collect_candidates,
    assign_successor_splits,
    freeze_corpus,
    freeze_successor_corpus,
    harden_candidate_sources,
    preview_successor_corpus,
    replay_successor_corpus,
)


def _write_transcript(
    path: Path,
    audio_path: Path,
    *,
    conversation_id: str,
    recording_id: str,
    speaker: str,
) -> Path:
    payload = {
        "schema_version": 2,
        "transcript_title": path.stem,
        "backend": "test",
        "conversation_id": conversation_id,
        "recording_id": recording_id,
        "recording_start": "2024-01-01T00:00:00Z",
        "recording_end": "2024-01-01T00:10:00Z",
        "duration_seconds": 600,
        "source_media_path": str(audio_path),
        "transcript_text": "fixture",
        "utterance_count": 2,
        "utterances": [
            {"speaker": speaker, "start": 0, "end": 5, "text": "first"},
            {"speaker": "B", "start": 4, "end": 8, "text": "second"},
        ],
        "event": None,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _build_fixture(
    tmp_path: Path,
    *,
    count: int = 2,
    all_shared_identity: bool = False,
) -> tuple[Path, Path, str]:
    store_root = tmp_path / "store"
    campaign_root = tmp_path / "campaigns"
    campaign_id = "campaign-" + "1" * 20
    campaign_dir = campaign_root / campaign_id
    gold_dir = campaign_dir / "gold"
    gold_dir.mkdir(parents=True)
    records = []
    manifest_items = []
    for index in range(1, count + 1):
        conversation_id = f"conversation-{index}"
        audio = tmp_path / f"audio-{index}.m4a"
        audio.write_bytes(f"audio fixture {index}".encode("utf-8"))
        transcript = _write_transcript(
            tmp_path / f"case-{index}.transcript.json",
            audio,
            conversation_id=conversation_id,
            recording_id=f"recording-{index}",
            speaker="A",
        )
        result = transcript_store.ingest_artifact(transcript, root=store_root)
        gold_id = f"gold-{index}"
        gold_path = gold_dir / f"{gold_id}.json"
        gold = {
            "schema_version": "transcribe-audio.speaker-evaluation-gold.v1",
            "gold_id": gold_id,
            "campaign_id": campaign_id,
            "manifest_id": "manifest-fixture",
            "document_id": result.id,
            "chronological_rank": index,
            "artifact_sha256": transcript_store.sha256_file(transcript),
            "disposition": "eligible_known",
            "review_method": "operator_confirmed_test_fixture",
            "reviewer": "Test Operator",
            "reviewed_at": "2026-07-31T00:00:00Z",
            "prediction_visibility": "excluded",
            "speaker_outcomes": [
                {
                    "speaker_label": "A",
                    "outcome": "known",
                    "person_ground_truth_id": "person-shared",
                },
                {
                    "speaker_label": "B",
                    "outcome": "known",
                    "person_ground_truth_id": (
                        "person-shared"
                        if all_shared_identity
                        else "person-secondary"
                        if index <= 2
                        else f"person-{index}"
                    ),
                },
            ],
            "same_person_label_groups": [],
        }
        gold_path.write_text(json.dumps(gold), encoding="utf-8")
        records.append(
            {
                "gold_id": gold_id,
                "document_id": result.id,
                "chronological_rank": index,
                "disposition": "eligible_known",
                "reviewed_at": "2026-07-31T00:00:00Z",
                "path": str(gold_path),
            }
        )
        manifest_items.append(
            {
                "document_id": result.id,
                "chronological_rank": index,
                    "artifact_sha256": transcript_store.sha256_file(transcript),
            }
        )
    (campaign_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": (
                    "transcribe-audio.speaker-evaluation-campaign-manifest.v1"
                ),
                "campaign_id": campaign_id,
                "manifest_id": "manifest-fixture",
                "items": manifest_items,
            }
        ),
        encoding="utf-8",
    )
    (gold_dir / "index.json").write_text(
        json.dumps(
            {
                "schema_version": (
                    "transcribe-audio.speaker-evaluation-gold-index.v1"
                ),
                "records": records,
            }
        ),
        encoding="utf-8",
    )
    return store_root, campaign_root, campaign_id


def _successor_authorities(
    tmp_path: Path,
    campaign_root: Path,
    campaign_id: str,
) -> tuple[Path, list[Path], dict[str, object]]:
    campaign_dir = campaign_root / campaign_id
    manifest = json.loads((campaign_dir / "manifest.json").read_text())
    index = json.loads((campaign_dir / "gold" / "index.json").read_text())
    freeze_dir = campaign_dir / "freezes"
    freeze_dir.mkdir()
    gold_freeze = freeze_dir / "freeze-fixture.json"
    gold_freeze.write_text(
        json.dumps(
            {
                "schema_version": (
                    "transcribe-audio.speaker-evaluation-gold-freeze.v1"
                ),
                "freeze_id": "freeze-fixture",
                "campaign_id": campaign_id,
                "manifest_id": manifest["manifest_id"],
                "status": "gold_batch_frozen",
                "prediction_visibility": "excluded",
                "gold_case_count": len(index["records"]),
                "document_ids": [item["document_id"] for item in index["records"]],
                "gold_ids": [item["gold_id"] for item in index["records"]],
            }
        ),
        encoding="utf-8",
    )
    gold_freeze.chmod(0o600)
    prior_paths = []
    for index in (1, 2):
        prior_path = tmp_path / f"prior-corpus-{index}.json"
        prior_path.write_text(
            json.dumps({"corpus_id": f"prior-fixture-{index}", "recordings": []}),
            encoding="utf-8",
        )
        prior_path.chmod(0o600)
        prior_paths.append(prior_path)
    module_path = Path(__file__).parents[1] / "acoustic_evaluation_corpus.py"
    repository_authority = {
        "commit": "a" * 40,
        "clean": True,
        "module_sha256": hashlib.sha256(module_path.read_bytes()).hexdigest(),
    }
    return gold_freeze, prior_paths, repository_authority


def test_corpus_freeze_is_private_disjoint_and_idempotent(tmp_path: Path) -> None:
    store_root, campaign_root, campaign_id = _build_fixture(tmp_path)
    runtime_root = tmp_path / "runtime"
    candidates, metadata = collect_candidates(
        campaign_id,
        campaign_root=campaign_root,
        store_root=store_root,
    )

    assert len(candidates) == 2
    assert all(
        item["operator_gold"]["prediction_visibility"] == "excluded"
        for item in candidates
    )
    Path(candidates[0]["source_blob"]["stored_path"]).chmod(0o644)
    with pytest.raises(CorpusError, match="not private"):
        freeze_corpus(
            candidates,
            metadata,
            runtime_root=runtime_root,
            approval_token=FREEZE_TOKEN,
        )

    hardening = harden_candidate_sources(
        candidates,
        store_root=store_root,
        approval_token=HARDEN_TOKEN,
    )
    assert hardening["content_modified"] is False
    candidates, metadata = collect_candidates(
        campaign_id,
        campaign_root=campaign_root,
        store_root=store_root,
    )
    first = freeze_corpus(
        candidates,
        metadata,
        runtime_root=runtime_root,
        approval_token=FREEZE_TOKEN,
    )
    second = freeze_corpus(
        candidates,
        metadata,
        runtime_root=runtime_root,
        approval_token=FREEZE_TOKEN,
    )

    assert first == second
    assert first["denominators"]["recordings"] == 2
    manifest_path = Path(first["manifest_path"])
    assert stat.S_IMODE(manifest_path.stat().st_mode) == 0o600
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    owners: dict[str, set[str]] = {}
    for item in manifest["recordings"]:
        owners.setdefault(item["conversation_id"], set()).add(item["split"])
    assert all(len(splits) == 1 for splits in owners.values())
    assert manifest["promotion_eligible"] is False
    assert manifest["prediction_visibility"] == "excluded"


def test_collect_rejects_non_operator_gold(tmp_path: Path) -> None:
    store_root, campaign_root, campaign_id = _build_fixture(tmp_path)
    gold_path = next((campaign_root / campaign_id / "gold").glob("gold-*.json"))
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    gold["review_method"] = "model_proposal"
    gold_path.write_text(json.dumps(gold), encoding="utf-8")

    with pytest.raises(CorpusError, match="operator-confirmed"):
        collect_candidates(
            campaign_id,
            campaign_root=campaign_root,
            store_root=store_root,
        )


def test_collect_rejects_gold_path_escape_and_identity_mismatch(
    tmp_path: Path,
) -> None:
    store_root, campaign_root, campaign_id = _build_fixture(tmp_path)
    index_path = campaign_root / campaign_id / "gold" / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    original_path = Path(index["records"][0]["path"])
    escaped_path = tmp_path / "escaped-gold.json"
    escaped_path.write_bytes(original_path.read_bytes())
    index["records"][0]["path"] = str(escaped_path)
    index_path.write_text(json.dumps(index), encoding="utf-8")

    with pytest.raises(CorpusError, match="escapes"):
        collect_candidates(
            campaign_id,
            campaign_root=campaign_root,
            store_root=store_root,
        )

    index["records"][0]["path"] = str(original_path)
    index_path.write_text(json.dumps(index), encoding="utf-8")
    gold = json.loads(original_path.read_text(encoding="utf-8"))
    gold["artifact_sha256"] = "f" * 64
    original_path.write_text(json.dumps(gold), encoding="utf-8")
    with pytest.raises(CorpusError, match="provenance do not match"):
        collect_candidates(
            campaign_id,
            campaign_root=campaign_root,
            store_root=store_root,
        )


def test_successor_preview_apply_and_replay_are_exact_and_private(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_root, campaign_root, campaign_id = _build_fixture(tmp_path, count=7)
    candidates, metadata = collect_candidates(
        campaign_id, campaign_root=campaign_root, store_root=store_root
    )
    harden_candidate_sources(
        candidates, store_root=store_root, approval_token=HARDEN_TOKEN
    )
    candidates, metadata = collect_candidates(
        campaign_id, campaign_root=campaign_root, store_root=store_root
    )
    gold_freeze, prior_paths, repository_authority = _successor_authorities(
        tmp_path, campaign_root, campaign_id
    )
    monkeypatch.setattr(
        corpus,
        "_current_repository_authority",
        lambda: dict(repository_authority),
    )

    preview = preview_successor_corpus(
        candidates,
        metadata,
        gold_freeze_path=gold_freeze,
        prior_corpus_paths=prior_paths,
        repository_authority=repository_authority,
    )
    assert preview["denominators"]["split_recordings"] == {
        "development": 3,
        "calibration": 2,
        "evaluation": 2,
    }
    assert preview["denominators"]["recordings"] == 7
    assert preview["denominators"]["subjects"] >= 5
    assert preview["denominators"]["recurrent_subjects"] >= 2
    assert preview["denominators"]["feasible_same_person_pairs"] >= 4
    assert preview["benchmark_readiness"] == {
        "status": "ready_for_p1_measurement",
        "blockers": [],
    }
    assert preview["promotion_eligible"] is False

    source_path = Path(candidates[0]["source_blob"]["stored_path"])
    source_bytes = source_path.read_bytes()
    source_path.write_bytes(source_bytes + b"drift")
    with pytest.raises(CorpusError, match="candidate authority drifted"):
        freeze_successor_corpus(
            candidates,
            metadata,
            gold_freeze_path=gold_freeze,
            prior_corpus_paths=prior_paths,
            repository_authority=repository_authority,
            runtime_root=tmp_path / "runtime",
            expected_content_sha256=preview["content_sha256"],
        )
    source_path.write_bytes(source_bytes)

    transcript_path = Path(
        candidates[0]["transcript_lineage"]["current_artifact_path"]
    )
    transcript_bytes = transcript_path.read_bytes()
    transcript_path.write_bytes(transcript_bytes + b" ")
    with pytest.raises(CorpusError, match="candidate authority drifted"):
        freeze_successor_corpus(
            candidates,
            metadata,
            gold_freeze_path=gold_freeze,
            prior_corpus_paths=prior_paths,
            repository_authority=repository_authority,
            runtime_root=tmp_path / "runtime",
            expected_content_sha256=preview["content_sha256"],
        )
    transcript_path.write_bytes(transcript_bytes)

    index_path = campaign_root / campaign_id / "gold" / "index.json"
    index_bytes = index_path.read_bytes()
    gold_path = Path(json.loads(index_bytes)["records"][0]["path"])
    gold_bytes = gold_path.read_bytes()
    changed_gold = json.loads(gold_bytes)
    changed_gold["speaker_outcomes"][0]["person_ground_truth_id"] = "person-drift"
    gold_path.write_text(json.dumps(changed_gold))
    with pytest.raises(CorpusError, match="candidate authority drifted"):
        freeze_successor_corpus(
            candidates,
            metadata,
            gold_freeze_path=gold_freeze,
            prior_corpus_paths=prior_paths,
            repository_authority=repository_authority,
            runtime_root=tmp_path / "runtime",
            expected_content_sha256=preview["content_sha256"],
        )
    gold_path.write_bytes(gold_bytes)

    changed_gold = json.loads(gold_bytes)
    changed_gold["review_method"] = "model_proposal"
    gold_path.write_text(json.dumps(changed_gold))
    with pytest.raises(CorpusError, match="candidate authority drifted"):
        freeze_successor_corpus(
            candidates,
            metadata,
            gold_freeze_path=gold_freeze,
            prior_corpus_paths=prior_paths,
            repository_authority=repository_authority,
            runtime_root=tmp_path / "runtime",
            expected_content_sha256=preview["content_sha256"],
        )
    gold_path.write_bytes(gold_bytes)

    index_path.write_bytes(index_bytes + b" ")
    with pytest.raises(CorpusError, match="gold-index authority drifted"):
        freeze_successor_corpus(
            candidates,
            metadata,
            gold_freeze_path=gold_freeze,
            prior_corpus_paths=prior_paths,
            repository_authority=repository_authority,
            runtime_root=tmp_path / "runtime",
            expected_content_sha256=preview["content_sha256"],
        )
    index_path.write_bytes(index_bytes)

    with pytest.raises(CorpusError, match="preview content hash is stale"):
        freeze_successor_corpus(
            candidates,
            metadata,
            gold_freeze_path=gold_freeze,
            prior_corpus_paths=prior_paths,
            repository_authority=repository_authority,
            runtime_root=tmp_path / "runtime",
            expected_content_sha256="f" * 64,
        )
    receipt = freeze_successor_corpus(
        candidates,
        metadata,
        gold_freeze_path=gold_freeze,
        prior_corpus_paths=prior_paths,
        repository_authority=repository_authority,
        runtime_root=tmp_path / "runtime",
        expected_content_sha256=preview["content_sha256"],
    )
    repeated = freeze_successor_corpus(
        candidates,
        metadata,
        gold_freeze_path=gold_freeze,
        prior_corpus_paths=prior_paths,
        repository_authority=repository_authority,
        runtime_root=tmp_path / "runtime",
        expected_content_sha256=preview["content_sha256"],
    )
    assert receipt == repeated
    assert stat.S_IMODE(Path(receipt["manifest_path"]).stat().st_mode) == 0o600
    manifest = json.loads(Path(receipt["manifest_path"]).read_text())
    assert manifest["terminal_selection_readiness"]["status"] == (
        "pending_condition_measurement"
    )
    assert manifest["promotion_eligible"] is False
    assert all(
        "unassessed_until_" in value
        for item in manifest["recordings"]
        for key, value in item["conditions"].items()
        if key != "overlap"
    )
    replay = replay_successor_corpus(
        Path(receipt["manifest_path"]),
        candidates,
        metadata,
        gold_freeze_path=gold_freeze,
        prior_corpus_paths=prior_paths,
        repository_authority=repository_authority,
    )
    assert replay["full_body_match"] is True
    manifest["runtime_readback_at_freeze"] = {"tampered": True}
    Path(receipt["manifest_path"]).write_text(json.dumps(manifest))
    with pytest.raises(CorpusError, match="manifest body does not match"):
        replay_successor_corpus(
            Path(receipt["manifest_path"]),
            candidates,
            metadata,
            gold_freeze_path=gold_freeze,
            prior_corpus_paths=prior_paths,
            repository_authority=repository_authority,
        )


def test_successor_rejects_wrong_denominator_overlap_and_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_root, campaign_root, campaign_id = _build_fixture(tmp_path, count=7)
    candidates, metadata = collect_candidates(
        campaign_id, campaign_root=campaign_root, store_root=store_root
    )
    harden_candidate_sources(
        candidates, store_root=store_root, approval_token=HARDEN_TOKEN
    )
    candidates, metadata = collect_candidates(
        campaign_id, campaign_root=campaign_root, store_root=store_root
    )
    gold_freeze, prior_paths, repository_authority = _successor_authorities(
        tmp_path, campaign_root, campaign_id
    )
    monkeypatch.setattr(
        corpus,
        "_current_repository_authority",
        lambda: dict(repository_authority),
    )

    insufficient_store, insufficient_campaign_root, insufficient_campaign_id = (
        _build_fixture(
            tmp_path / "insufficient",
            count=7,
            all_shared_identity=True,
        )
    )
    insufficient, insufficient_metadata = collect_candidates(
        insufficient_campaign_id,
        campaign_root=insufficient_campaign_root,
        store_root=insufficient_store,
    )
    harden_candidate_sources(
        insufficient,
        store_root=insufficient_store,
        approval_token=HARDEN_TOKEN,
    )
    insufficient, insufficient_metadata = collect_candidates(
        insufficient_campaign_id,
        campaign_root=insufficient_campaign_root,
        store_root=insufficient_store,
    )
    (
        insufficient_gold_freeze,
        insufficient_prior_paths,
        insufficient_repository_authority,
    ) = _successor_authorities(
        tmp_path / "insufficient",
        insufficient_campaign_root,
        insufficient_campaign_id,
    )
    insufficient_preview = preview_successor_corpus(
        insufficient,
        insufficient_metadata,
        gold_freeze_path=insufficient_gold_freeze,
        prior_corpus_paths=insufficient_prior_paths,
        repository_authority=insufficient_repository_authority,
    )
    assert insufficient_preview["benchmark_readiness"]["status"] == "insufficient"
    rejected_runtime = tmp_path / "rejected-runtime"
    with pytest.raises(CorpusError, match="evidence-feasibility gates"):
        freeze_successor_corpus(
            insufficient,
            insufficient_metadata,
            gold_freeze_path=insufficient_gold_freeze,
            prior_corpus_paths=insufficient_prior_paths,
            repository_authority=insufficient_repository_authority,
            runtime_root=rejected_runtime,
            expected_content_sha256=insufficient_preview["content_sha256"],
        )
    assert not rejected_runtime.exists()
    with pytest.raises(CorpusError, match="exactly seven"):
        assign_successor_splits(candidates[:6])
    with pytest.raises(CorpusError, match="exactly two prior corpora"):
        preview_successor_corpus(
            candidates,
            metadata,
            gold_freeze_path=gold_freeze,
            prior_corpus_paths=prior_paths[:1],
            repository_authority=repository_authority,
        )

    dirty_authority = {**repository_authority, "clean": False}
    monkeypatch.setattr(
        corpus,
        "_current_repository_authority",
        lambda: dict(dirty_authority),
    )
    with pytest.raises(CorpusError, match="repository authority"):
        preview_successor_corpus(
            candidates,
            metadata,
            gold_freeze_path=gold_freeze,
            prior_corpus_paths=prior_paths,
            repository_authority=repository_authority,
        )
    monkeypatch.setattr(
        corpus,
        "_current_repository_authority",
        lambda: dict(repository_authority),
    )

    prior = json.loads(prior_paths[0].read_text())
    prior["recordings"] = [
        {
            "document_id": "prior-document",
            "recording_id": candidates[0]["recording_id"],
            "conversation_id": "prior-conversation",
            "source_blob": {"sha256": "e" * 64},
        }
    ]
    prior_paths[0].write_text(json.dumps(prior), encoding="utf-8")
    with pytest.raises(CorpusError, match="overlaps prior corpus"):
        preview_successor_corpus(
            candidates,
            metadata,
            gold_freeze_path=gold_freeze,
            prior_corpus_paths=prior_paths,
                repository_authority=repository_authority,
        )

    prior["recordings"] = []
    prior_paths[0].write_text(json.dumps(prior), encoding="utf-8")
    preview = preview_successor_corpus(
        candidates,
        metadata,
        gold_freeze_path=gold_freeze,
        prior_corpus_paths=prior_paths,
        repository_authority=repository_authority,
    )
    receipt = freeze_successor_corpus(
        candidates,
        metadata,
        gold_freeze_path=gold_freeze,
        prior_corpus_paths=prior_paths,
        repository_authority=repository_authority,
        runtime_root=tmp_path / "runtime",
        expected_content_sha256=preview["content_sha256"],
    )
    manifest_path = Path(receipt["manifest_path"])
    manifest = json.loads(manifest_path.read_text())
    manifest["selection_policy"]["split_algorithm"] = "tampered"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(CorpusError, match="manifest body does not match"):
        replay_successor_corpus(
            manifest_path,
            candidates,
            metadata,
            gold_freeze_path=gold_freeze,
            prior_corpus_paths=prior_paths,
            repository_authority=repository_authority,
        )
