#!/usr/bin/env python3
"""Serve a disposable redacted Plan 0072 A5 UI fixture without live effects."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
import wave
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import transcript_api
import transcript_store
from conversation_knowledge_store import ConversationKnowledgeStore
from identity_review_workflow import IdentityReviewWorkflow


FIXTURE_ROOT = REPO_ROOT / "docs" / "dev" / "fixtures" / "plan-0072-a5"


def read_object(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Fixture is not an object: {path}.")
    return payload


def write_tone(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    sample_rate = 16_000
    duration_seconds = 8
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        frames = bytearray()
        for index in range(sample_rate * duration_seconds):
            amplitude = int(4_200 * math.sin(2 * math.pi * 220 * index / sample_rate))
            frames.extend(struct.pack("<h", amplitude))
        output.writeframes(bytes(frames))
    path.chmod(0o600)


def prepare_fixture(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    root.chmod(0o700)
    source_root = root / "fixture-source"
    source_root.mkdir(exist_ok=True, mode=0o700)
    source_root.chmod(0o700)
    media_path = source_root / "Monday planning.wav"
    transcript_path = source_root / "Monday planning.transcript.json"
    if not media_path.is_file():
        write_tone(media_path)
    transcript_payload = {
        "transcript_title": "Monday planning",
        "transcript_text": "SPEAKER_01 [1.25s - 6.90s]: This redacted fixture proves source-bound audio controls.",
        "source_media_path": str(media_path),
        "working_media_path": str(media_path),
        "backend": "plan0072-a5-redacted-fixture",
        "duration_seconds": 8,
        "utterances": [
            {
                "speaker": "SPEAKER_01",
                "start": 1250,
                "end": 6900,
                "text": "This redacted fixture proves source-bound audio controls.",
            }
        ],
    }
    transcript_path.write_text(
        json.dumps(transcript_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    transcript_path.chmod(0o600)
    result = transcript_store.ingest_artifact(
        transcript_path,
        root=root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash-v1",
    )
    ConversationKnowledgeStore(root).migrate(backup=False)
    with transcript_store.connect(root) as con:
        blob = con.execute(
            """
            SELECT blobs.id, blobs.sha256
            FROM blobs
            JOIN document_blobs ON document_blobs.blob_id = blobs.id
            WHERE document_blobs.document_id = ? AND document_blobs.role = 'source_recording'
            """,
            (result.id,),
        ).fetchone()
    if blob is None:
        raise RuntimeError("Redacted preview media blob was not registered.")
    queue = read_object(FIXTURE_ROOT / "identity-review-queue-item.json")
    queue.update(
        {
            "source_artifact_sha256": hashlib.sha256(transcript_path.read_bytes()).hexdigest(),
            "source_media_sha256": str(blob["sha256"]),
        }
    )
    queue["speakers"][0]["audio"]["media_url"] = f"/api/blobs/{blob['id']}"
    IdentityReviewWorkflow(root).project_queue_item(
        queue,
        priority=90,
        impact_score=0.8,
    )
    people = read_object(FIXTURE_ROOT / "people-projection.json")
    with transcript_store.connect(root) as con:
        con.execute(
            """
            INSERT OR REPLACE INTO knowledge_identity_people_projection (
              person_id, status, primary_name, aliases_json, merged_into_person_id,
              input_watermark, metadata_json, built_at
            ) VALUES (?, ?, ?, ?, ?, ?, '{}', ?)
            """,
            (
                people["person_id"], people["status"], people["primary_name"],
                json.dumps(people["aliases"]), people["merged_into_person_id"],
                people["input_watermark"], people["built_at"],
            ),
        )
        for source in people["source_records"]:
            con.execute(
                """
                INSERT OR REPLACE INTO knowledge_identity_source_projection (
                  source_record_id, person_id, source_profile_id, provider_kind,
                  account_id, tenant_id, record_type, external_ref, label,
                  source_event_at, observed_at, content_hash, resolution_status,
                  input_watermark, metadata_json, built_at
                ) VALUES (?, ?, 'plan0072-a5-fixture', ?, 'fixture-account',
                  'fixture-tenant', ?, ?, ?, '', ?, ?, ?, ?, '{}', ?)
                """,
                (
                    source["source_record_id"], people["person_id"],
                    source["provider_kind"], source["record_type"],
                    source["source_record_id"], source["label"], people["built_at"],
                    hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest(),
                    source["resolution_status"], people["input_watermark"], people["built_at"],
                ),
            )
        for role in people["roles"]:
            con.execute(
                """
                INSERT OR REPLACE INTO knowledge_identity_role_projection (
                  role_id, person_id, role_type, organization_id, project_id,
                  matter_id, conversation_id, starts_at, ends_at, status,
                  evidence_ids_json, input_watermark, metadata_json, built_at
                ) VALUES (?, ?, ?, ?, '', '', '', '', '', ?, '[]', ?, '{}', ?)
                """,
                (
                    role["role_id"], people["person_id"], role["role_type"],
                    role["organization_id"], role["status"],
                    people["input_watermark"], people["built_at"],
                ),
            )
        for relationship in people["relationships"]:
            con.execute(
                """
                INSERT OR REPLACE INTO knowledge_identity_relationship_projection (
                  relationship_id, relationship_type, subject_type, subject_id,
                  object_type, object_id, directionality, inverse_relationship_id,
                  starts_at, ends_at, status, evidence_ids_json, input_watermark,
                  metadata_json, built_at
                ) VALUES (?, ?, 'person', ?, 'person', ?, 'symmetric', '', '', '',
                  ?, '[]', ?, '{}', ?)
                """,
                (
                    relationship["relationship_id"], relationship["relationship_type"],
                    relationship["subject_id"], relationship["object_id"],
                    relationship["status"], people["input_watermark"], people["built_at"],
                ),
            )
        con.commit()
    return {
        "schema_version": "transcribe-audio.plan0072-a5-preview.v1",
        "status": "ready",
        "store_root": str(root),
        "document_id": result.id,
        "queue_item_id": queue["queue_item_id"],
        "original_recording_filename": queue["original_recording_filename"],
        "provider_call_count": 0,
        "provider_write_count": 0,
        "accepted_identity_effect_count": 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18972)
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    receipt = prepare_fixture(root)
    print("PLAN0072_A5_PREVIEW_JSON=" + json.dumps(receipt, sort_keys=True), flush=True)
    if args.prepare_only:
        return 0
    server = transcript_api.TranscriptApiServer(
        (args.host, args.port),
        transcript_api.TranscriptApiHandler,
        store_root=root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash-v1",
        state_root=root / "state",
        quiet=True,
        static_dir=REPO_ROOT / "frontend" / "dist",
    )
    print(f"PLAN0072_A5_PREVIEW_URL=http://{args.host}:{args.port}/?view=Identity+Review", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 130
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
