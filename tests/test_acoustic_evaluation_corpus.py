from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

import transcript_store
from acoustic_evaluation_corpus import (
    FREEZE_TOKEN,
    HARDEN_TOKEN,
    CorpusError,
    collect_candidates,
    freeze_corpus,
    harden_candidate_sources,
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


def _build_fixture(tmp_path: Path) -> tuple[Path, Path, str]:
    store_root = tmp_path / "store"
    campaign_root = tmp_path / "campaigns"
    campaign_id = "campaign-" + "1" * 20
    campaign_dir = campaign_root / campaign_id
    gold_dir = campaign_dir / "gold"
    gold_dir.mkdir(parents=True)
    records = []
    manifest_items = []
    for index, conversation_id in enumerate(("conversation-a", "conversation-b"), 1):
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
                    "person_ground_truth_id": f"person-{index}",
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
