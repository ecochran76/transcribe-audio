from __future__ import annotations

import hashlib
import json
import stat
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import conversation_knowledge_projection
import conversation_knowledge_store
import transcript_store


CONVERSATION_ID = "00000000-0000-4000-8000-000000000101"
RECORDING_ID = "00000000-0000-4000-8000-000000000102"
EVALUATION_ID = "00000000-0000-4000-8000-000000000103"
DECISION_ID = "00000000-0000-4000-8000-000000000104"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_projection_fixture(tmp_path: Path) -> tuple[Path, Path, str]:
    transcript_path = tmp_path / "projection.transcript.json"
    transcript_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "conversation_id": CONVERSATION_ID,
                "recording_id": RECORDING_ID,
                "transcript_title": "Projection fixture",
                "backend": "assembly",
                "model": "universal-3-pro",
                "recording_start": "2026-07-26T09:00:00-05:00",
                "recording_end": "2026-07-26T09:30:00-05:00",
                "duration_seconds": 1800,
                "transcript_text": "Hello. Project update.",
                "utterances": [
                    {
                        "speaker": "A",
                        "start": 0,
                        "end": 500,
                        "text": "Hello.",
                    },
                    {
                        "speaker": "B",
                        "start": 500,
                        "end": 1500,
                        "text": "Project update.",
                    },
                ],
                "event": {"id": "calendar-event-1", "summary": "Projection fixture"},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    sidecar_path = tmp_path / "projection.processing.json"
    sidecar_path.write_text(
        json.dumps(
            {
                "schema_version": "transcribe-audio.conversation-processing.v1",
                "conversation_id": CONVERSATION_ID,
                "recording_ids": [RECORDING_ID],
                "current_evaluation_id": EVALUATION_ID,
                "evaluations": [
                    {
                        "evaluation_id": EVALUATION_ID,
                        "evaluation_type": "speaker_identity",
                        "schema_version": "speaker-identity.v1",
                        "status": "complete",
                        "created_at": "2026-07-26T15:00:00Z",
                        "model_profile": "codex-app-server",
                        "proposals": [
                            {
                                "proposal_id": "proposal-a",
                                "speaker_labels": ["A"],
                                "identity": {"name": "Example Person"},
                            }
                        ],
                    }
                ],
                "review_decisions": [
                    {
                        "decision_id": DECISION_ID,
                        "evaluation_id": EVALUATION_ID,
                        "proposal_id": "proposal-a",
                        "action": "confirm",
                        "reviewer": "operator",
                        "decision_method": "manual",
                        "decided_at": "2026-07-26T15:05:00Z",
                        "reviewer_note": "Confirmed from conversation context.",
                        "supersedes_decision_id": "",
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    ingested = transcript_store.ingest_artifact(
        transcript_path,
        root=tmp_path,
        embedding_provider="debug-hash",
        embedding_model="debug-hash-v1",
    )
    now = "2026-07-26T15:10:00Z"
    with transcript_store.connect(tmp_path) as con:
        transcript_store.init_db(con)
        con.execute(
            """
            INSERT INTO contacts (
                id, label, email, external_ref, metadata_json,
                created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy-contact-1",
                "Example Person",
                "person@example.com",
                "people/1",
                json.dumps({"source": "operator_created"}),
                now,
                now,
            ),
        )
        con.execute(
            """
            INSERT INTO speaker_assignments (
                id, conversation_key, document_id, speaker_label,
                contact_id, contact_label, status, confidence,
                evidence_json, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy-assignment-1",
                str(transcript_path.resolve()),
                ingested.id,
                "A",
                "legacy-contact-1",
                "Example Person",
                "confirmed",
                1.0,
                json.dumps([{"source": "operator_review"}]),
                now,
                now,
            ),
        )
        con.commit()
    return transcript_path, sidecar_path, ingested.id


def test_preview_is_read_only_and_apply_round_trips_sidecar(
    tmp_path: Path,
) -> None:
    transcript_path, sidecar_path, document_id = _write_projection_fixture(
        tmp_path
    )
    transcript_hash = _sha256(transcript_path)
    sidecar_hash = _sha256(sidecar_path)
    projector = conversation_knowledge_projection.ConversationKnowledgeProjector(
        tmp_path
    )

    plan = projector.preview(
        transcript_path,
        document_id=document_id,
    )

    assert plan.authority_mode == "sidecar"
    assert plan.source_transcript_sha256 == transcript_hash
    assert plan.source_sidecar_sha256 == sidecar_hash
    assert plan.counts == {
        "assignments": 1,
        "contacts": 1,
        "conversations": 1,
        "decisions": 1,
        "evaluations": 1,
        "proposals": 1,
        "recordings": 1,
        "utterances": 2,
    }
    assert conversation_knowledge_store.ConversationKnowledgeStore(
        tmp_path
    ).schema_status().schema_version == 0
    assert _sha256(transcript_path) == transcript_hash
    assert _sha256(sidecar_path) == sidecar_hash

    with pytest.raises(ValueError, match="approval token"):
        projector.apply(plan, approval_token="wrong")

    first = projector.apply(
        plan,
        approval_token=conversation_knowledge_projection.APPLY_APPROVAL_TOKEN,
        migrate_backup=False,
    )
    receipt_hash = _sha256(Path(first.receipt_path))
    second = projector.apply(
        plan,
        approval_token=conversation_knowledge_projection.APPLY_APPROVAL_TOKEN,
        migrate_backup=False,
    )

    assert first.status == "inserted"
    assert first.reconciled is True
    assert second.status == "unchanged"
    assert second.reconciled is True
    assert _sha256(Path(second.receipt_path)) == receipt_hash
    assert stat.S_IMODE(Path(first.receipt_path).stat().st_mode) == 0o600
    assert stat.S_IMODE(Path(first.receipt_path).parent.stat().st_mode) == 0o700
    assert first.input_watermark == plan.input_watermark
    assert _sha256(transcript_path) == transcript_hash
    assert _sha256(sidecar_path) == sidecar_hash

    export_path = tmp_path / "exports" / "projection.processing.json"
    exported = projector.export_sidecar(CONVERSATION_ID, export_path)
    assert exported == json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert json.loads(export_path.read_text(encoding="utf-8")) == exported
    assert _sha256(sidecar_path) == sidecar_hash

    store = conversation_knowledge_store.ConversationKnowledgeStore(tmp_path)
    assert (
        store.load_conversation_snapshot(CONVERSATION_ID)
        == plan.conversation_snapshot
    )
    assert store.load_processing_history(CONVERSATION_ID) == plan.processing_history
    assert len(store.load_observations(CONVERSATION_ID)) == 1
    projection_state = store.load_projection_state(
        "sidecar-shadow",
        "conversation",
        CONVERSATION_ID,
    )
    assert projection_state is not None
    assert projection_state.input_watermark == plan.input_watermark


def test_apply_refuses_changed_sources(tmp_path: Path) -> None:
    transcript_path, sidecar_path, document_id = _write_projection_fixture(
        tmp_path
    )
    projector = conversation_knowledge_projection.ConversationKnowledgeProjector(
        tmp_path
    )
    plan = projector.preview(transcript_path, document_id=document_id)
    sidecar_path.write_text(
        sidecar_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="changed after preview"):
        projector.apply(
            plan,
            approval_token=conversation_knowledge_projection.APPLY_APPROVAL_TOKEN,
            migrate_backup=False,
        )
