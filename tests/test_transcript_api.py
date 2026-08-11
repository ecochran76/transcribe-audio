from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app_intelligence_ledger
import acoustic_shadow_evidence
import participant_identity
import transcript_api
import transcript_store
from routing_artifacts import ProvenanceSource


class FakeAuraCallHandler(BaseHTTPRequestHandler):
    requests: list[dict] = []

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        self.__class__.requests.append(payload)
        self.write_json(
            {
                "id": "batch_test",
                "status": "queued",
                "request_count": len(payload.get("requests") or []),
            },
            HTTPStatus.CREATED,
        )

    def do_GET(self) -> None:
        if self.path.endswith("/config/agent-choices"):
            self.write_json(
                {
                    "agents": [
                        {
                            "id": "transcripts-worker",
                            "label": "Transcripts worker",
                            "bindingKey": "binding:chatgpt:wsl-chrome-3:default",
                            "runtimeProfileId": "wsl-chrome-3",
                            "browserProfileId": "default",
                            "projectBinding": {"mode": "fixed", "label": "Transcripts"},
                        }
                    ],
                    "bindings": [
                        {
                            "bindingKey": "binding:chatgpt:wsl-chrome-3:default",
                            "ready": True,
                            "runtimeProfileId": "wsl-chrome-3",
                            "browserProfileId": "default",
                        }
                    ],
                    "teams": [],
                    "validation": {"agents": [{"agentId": "transcripts-worker", "valid": True}]},
                }
            )
            return
        if self.path.endswith("/response-batches/batch_test"):
            self.write_json(
                {
                    "id": "batch_test",
                    "status": "running",
                    "jobs": [{"index": 0, "status": "running"}],
                }
            )
            return
        self.write_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def write_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def write_transcript_artifact(tmp_path: Path) -> Path:
    media_path = tmp_path / "meeting.m4a"
    media_path.write_bytes(b"0123456789abcdef")
    artifact_path = tmp_path / "meeting.transcript.json"
    artifact_path.write_text(
        json.dumps(
            {
                "transcript_title": "Weekly Product Sync",
                "transcript_text": "Speaker A [0.00s - 1.00s]: We discussed Tempo Chemical samples.",
                "source_media_path": str(media_path),
                "working_media_path": str(media_path),
                "backend": "test",
                "duration_seconds": 16,
                "legacy_import": {
                    "needs_enrichment": True,
                    "source_path": str(artifact_path),
                    "source_sha256": "test-transcript-sha",
                },
                "utterances": [
                    {
                        "speaker": "Speaker A",
                        "start": 0,
                        "end": 1000,
                        "text": "We discussed Tempo Chemical samples.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return artifact_path


def add_calendar_attendee_context(artifact_path: Path) -> None:
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    payload["event"] = {
        "summary": "Weekly Product Sync",
        "participants": ["Alice Example <alice@example.com>"],
        "matching_calendars": [
            {
                "calendar_summary": "Work",
                "event_summary": "Weekly Product Sync",
                "attendees": [{"displayName": "Alice Example", "email": "alice@example.com"}],
            }
        ],
    }
    artifact_path.write_text(json.dumps(payload), encoding="utf-8")


def write_readout_artifact(tmp_path: Path, source_artifact: Path) -> Path:
    artifact_path = tmp_path / "meeting.readout.json"
    artifact_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "title": "Weekly Product Sync readout",
                "summary": "Tempo Chemical sample follow-up needs owner review.",
                "source_artifact_path": str(source_artifact.resolve()),
                "generated_at": "2026-05-22T12:00:00Z",
                "participants": ["Speaker A"],
                "topics": ["Tempo Chemical"],
                "action_items": [],
                "risks": [],
            }
        ),
        encoding="utf-8",
    )
    return artifact_path


def write_contextual_readout_artifact(tmp_path: Path, source_artifact: Path, route_path: Path) -> Path:
    artifact_path = tmp_path / "meeting.contextual.readout.json"
    artifact_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "title": "Weekly Product Sync contextual readout",
                "summary": "Tempo Chemical sample follow-up is tied to a reviewed matter route.",
                "source_artifact_path": str(source_artifact.resolve()),
                "generated_at": "2026-05-22T12:30:00Z",
                "participants": ["Speaker A"],
                "topics": ["Tempo Chemical"],
                "action_items": ["Send sample follow-up."],
                "risks": ["Route confidence needs review."],
                "memory_candidates": [
                    {
                        "kind": "relationship_context",
                        "text": "Tempo Chemical sample follow-up is associated with the product sync.",
                        "evidence": "Contextual readout and route source agree.",
                    }
                ],
                "contextualization": {
                    "route_status": "selected",
                    "excluded_source_count": 1,
                    "warnings": ["Excluded 1 provenance source below quality threshold 2."],
                    "selected_candidate": {
                        "label": "Tempo Chemical follow-up",
                        "target_kind": "matter",
                        "target_id": "matter-tempo",
                        "confidence": 0.82,
                    },
                    "supporting_context_sources": [
                        {
                            "source_id": "calendar-1",
                            "source_type": "calendar_event",
                            "label": "Weekly Product Sync",
                            "snippet": "Tempo Chemical samples.",
                        }
                    ],
                },
                "provider": {"route_decision_path": str(route_path)},
            }
        ),
        encoding="utf-8",
    )
    return artifact_path


def write_route_artifact(tmp_path: Path, source_artifact: Path, readout_artifact: Path) -> Path:
    route_path = tmp_path / "meeting.route.json"
    route_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "selected",
                "source_transcript_path": str(source_artifact.resolve()),
                "source_readout_path": str(readout_artifact.resolve()),
                "selected_candidate": {
                    "label": "Tempo Chemical follow-up",
                    "target_kind": "matter",
                    "target_id": "matter-tempo",
                    "confidence": 0.82,
                },
                "warnings": ["Excluded 1 provenance source below quality threshold 2."],
                "provenance_pack": {
                    "sources": [
                        {
                            "source_id": "calendar-1",
                            "source_type": "calendar_event",
                            "label": "Weekly Product Sync",
                            "snippet": "Tempo Chemical samples.",
                        }
                    ],
                    "excluded_sources": [
                        {
                            "source_id": "graphiti-noise",
                            "source_type": "graphiti_fact",
                            "label": "Unrelated advisory fact",
                            "snippet": "No useful overlap.",
                            "metadata": {"quality_status": "excluded_low_quality", "quality_score": 0},
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    return route_path


def test_ingest_registers_media_blob_for_api(tmp_path: Path) -> None:
    store_root = tmp_path / "store"
    result = transcript_store.ingest_artifact(
        write_transcript_artifact(tmp_path),
        root=store_root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
    )

    payload = transcript_api.get_document(result.id, root=store_root)

    assert payload["title"] == "Weekly Product Sync"
    assert payload["media_blob"]["id"]
    assert payload["media_blob"]["playback_url"].startswith("/api/blobs/")
    assert payload["blobs"][0]["bytes"] == 16
    assert Path(payload["metadata"]["media_blob"]["id"])


def test_related_documents_links_readout_to_source_transcript(tmp_path: Path) -> None:
    store_root = tmp_path / "store"
    transcript_path = write_transcript_artifact(tmp_path)
    transcript = transcript_store.ingest_artifact(
        transcript_path,
        root=store_root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
    )
    readout = transcript_store.ingest_artifact(
        write_readout_artifact(tmp_path, transcript_path),
        root=store_root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
    )

    readout_related = transcript_api.get_related_documents(readout.id, root=store_root)
    transcript_related = transcript_api.get_related_documents(transcript.id, root=store_root)

    assert readout_related["source_document"]["id"] == transcript.id
    assert readout_related["source_document"]["media_blob"]["playback_url"].startswith("/api/blobs/")
    assert readout_related["derived_documents"] == []
    assert transcript_related["source_document"] is None
    assert transcript_related["derived_documents"][0]["id"] == readout.id


def test_list_conversations_groups_transcript_readout_and_media(tmp_path: Path) -> None:
    store_root = tmp_path / "store"
    transcript_path = write_transcript_artifact(tmp_path)
    transcript = transcript_store.ingest_artifact(
        transcript_path,
        root=store_root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
    )
    readout = transcript_store.ingest_artifact(
        write_readout_artifact(tmp_path, transcript_path),
        root=store_root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
    )

    payload = transcript_api.list_conversations(root=store_root, limit=10)
    filtered = transcript_api.list_conversations(root=store_root, query="Tempo", limit=10)

    assert payload["schema_version"] == "transcribe-audio.conversations.v1"
    assert payload["total"] == 1
    conversation = payload["items"][0]
    assert conversation["representative"]["id"] == readout.id
    assert conversation["source"]["id"] == transcript.id
    assert conversation["workflow"] == {"transcript": True, "summary": True, "contextual_readout": False}
    assert conversation["media_ready"] is True
    assert conversation["media_blob"]["playback_url"].startswith("/api/blobs/")
    assert {artifact["id"] for artifact in conversation["artifacts"]} == {transcript.id, readout.id}
    assert filtered["total"] == 1


def test_get_conversation_detail_returns_transcript_summary_and_media(tmp_path: Path) -> None:
    store_root = tmp_path / "store"
    transcript_path = write_transcript_artifact(tmp_path)
    transcript = transcript_store.ingest_artifact(
        transcript_path,
        root=store_root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
    )
    readout = transcript_store.ingest_artifact(
        write_readout_artifact(tmp_path, transcript_path),
        root=store_root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
    )

    payload = transcript_api.get_conversation_detail(readout.id, root=store_root)

    assert payload["schema_version"] == "transcribe-audio.conversation-detail.v1"
    assert payload["conversation"]["representative"]["id"] == readout.id
    assert payload["selected_document"]["id"] == readout.id
    assert payload["transcript_document"]["id"] == transcript.id
    assert payload["summary_document"]["id"] == readout.id
    assert payload["contextual_readout_document"] is None
    assert payload["summary_document"]["json_payload"]["summary"].startswith("Tempo Chemical")
    assert payload["transcript_document"]["text_content"]
    assert payload["media_blob"]["playback_url"].startswith("/api/blobs/")
    assert payload["will_read_artifact_files"] is False
    assert payload["will_return_artifact_content"] is True


def test_conversation_detail_includes_identity_and_context_state(tmp_path: Path) -> None:
    store_root = tmp_path / "store"
    state_root = tmp_path / "state"
    transcript_path = write_transcript_artifact(tmp_path)
    readout_path = write_readout_artifact(tmp_path, transcript_path)
    route_path = write_route_artifact(tmp_path, transcript_path, readout_path)
    transcript = transcript_store.ingest_artifact(
        transcript_path,
        root=store_root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
    )
    transcript_store.ingest_artifact(
        readout_path,
        root=store_root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
    )
    contextual = transcript_store.ingest_artifact(
        write_contextual_readout_artifact(tmp_path, transcript_path, route_path),
        root=store_root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
    )

    payload = transcript_api.get_conversation_detail(contextual.id, root=store_root, state_root=state_root)

    assert payload["transcript_document"]["id"] == transcript.id
    assert payload["identity_review"]["pending_count"] == 1
    assert payload["identity_review"]["speakers"][0]["speaker_label"] == "Speaker A"
    assert payload["identity_review"]["joined_shadow_evidence"]["status"] == "absent"
    assert payload["identity_review"]["joined_shadow_evidence"]["apply_enabled"] is False
    assert payload["context_workbench"]["status"] == "contextual_readout_ready"
    assert payload["context_workbench"]["selected_candidate"]["label"] == "Tempo Chemical follow-up"
    assert payload["context_workbench"]["included_source_count"] == 1
    assert payload["context_workbench"]["excluded_source_count"] == 1
    assert payload["first_pass_summary"]["status"] == "summary_ready"
    assert payload["first_pass_summary"]["summary_document_id"]
    assert payload["review_state"]["context_status"] == "contextual_readout_ready"


def test_conversation_detail_adds_read_only_acoustic_shadow_evidence(
    tmp_path: Path,
) -> None:
    store_root = tmp_path / "store"
    state_root = tmp_path / "state"
    transcript = transcript_store.ingest_artifact(
        write_transcript_artifact(tmp_path),
        root=store_root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
    )
    initial = transcript_api.get_conversation_detail(
        transcript.id,
        root=store_root,
        state_root=state_root,
    )
    review = initial["identity_review"]
    assert review["acoustic_shadow_evidence"]["status"] == "absent"
    initial_fingerprint = review["identity_cache"]["fingerprint"]
    source_document = initial["transcript_document"]
    with transcript_api.connect(store_root) as con:
        before = {
            "contacts": con.execute("SELECT count(*) FROM contacts").fetchone()[0],
            "assignments": con.execute(
                "SELECT count(*) FROM speaker_assignments"
            ).fetchone()[0],
        }

    bundle = acoustic_shadow_evidence.build_shadow_bundle(
        document_id=source_document["id"],
        conversation_key=initial["conversation"]["key"],
        source_path=source_document["source_path"],
        source_media_sha256="a" * 64,
        execution_content_sha256="b" * 64,
        identity_state_sha256="c" * 64,
        rows=[
            {
                "speaker_ref": "SPEAKER_1",
                "disposition": "review",
                "subject_id": "subject-df34bc192c07bd86566fff12",
                "confidence_band": "low",
                "supporting_unit_count": 1,
                "supporting_candidate_family_count": 1,
                "opposing_unit_count": 0,
                "rationale": "Frozen consensus evidence.",
            }
        ],
    )
    acoustic_shadow_evidence.publish_shadow_bundle(
        bundle,
        source_path=source_document["source_path"],
        state_root=state_root,
    )

    refreshed = transcript_api.get_conversation_detail(
        transcript.id,
        root=store_root,
        state_root=state_root,
    )
    shadow = refreshed["identity_review"]["acoustic_shadow_evidence"]
    assert shadow["status"] == "available"
    assert shadow["content_sha256"] == bundle["content_sha256"]
    assert shadow["rows"][0]["subject_id"] == "subject-df34bc192c07bd86566fff12"
    assert shadow["will_apply_speaker_assignments"] is False
    assert refreshed["identity_review"]["identity_cache"]["fingerprint"] != initial_fingerprint
    with transcript_api.connect(store_root) as con:
        after = {
            "contacts": con.execute("SELECT count(*) FROM contacts").fetchone()[0],
            "assignments": con.execute(
                "SELECT count(*) FROM speaker_assignments"
            ).fetchone()[0],
        }
    assert after == before


def test_conversation_identity_bundle_uses_configured_contact_provenance(tmp_path: Path, monkeypatch) -> None:
    store_root = tmp_path / "store"
    state_root = tmp_path / "state"
    state_root.mkdir()
    (state_root / participant_identity.CONTACT_SOURCE_CONFIG_NAME).write_text(
        json.dumps({"gws": {"profiles": [{"label": "work", "surfaces": ["contacts"]}]}}),
        encoding="utf-8",
    )
    transcript_path = write_transcript_artifact(tmp_path)
    add_calendar_attendee_context(transcript_path)

    def fake_gws(query_terms, *, config):
        assert "alice@example.com" in query_terms
        return [
            ProvenanceSource(
                source_type="gws_contact",
                source_id="people/alice",
                label="Alice Example",
                snippet="Alice Example; alice@example.com",
                metadata={"profile": "work", "email": "alice@example.com"},
            )
        ]

    monkeypatch.setattr(participant_identity, "collect_gws_contact_provenance", fake_gws)
    transcript = transcript_store.ingest_artifact(
        transcript_path,
        root=store_root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
    )

    payload = transcript_api.get_conversation_detail(transcript.id, root=store_root, state_root=state_root)
    bundle = payload["identity_review"]["identity_bundle"]

    assert bundle["calendar_attendees"][0]["email"] == "alice@example.com"
    assert bundle["contact_candidates"][0]["source_type"] == "gws_contact"
    assert bundle["contact_candidates"][0]["confidence"] == 0.95
    assert payload["context_workbench"]["participant_identity_bundle"]["schema_version"]
    assert payload["context_workbench"]["proposed_contact_candidates"][0]["label"] == "Alice Example"
    assert payload["context_workbench"]["contact_selection"]["status"] == "review_needed"
    selection = transcript_api.record_context_contact_selection(
        transcript.id,
        root=store_root,
        state_root=state_root,
        candidate_id=bundle["contact_candidates"][0]["contact_id"],
        action="select",
        actor_type="app_intelligence",
        reviewer="app-intelligence-test",
        note="Selected by structured decision for context preview.",
    )
    assert selection["status"] == "select"
    assert selection["decision"]["actor_type"] == "app_intelligence"
    selected_state = selection["context_workbench"]["contact_selection"]
    assert selected_state["status"] == "selected"
    assert selected_state["selected_candidates"][0]["label"] == "Alice Example"
    assert Path(selected_state["selection_path"]).exists()
    manual_selection = transcript_api.record_context_contact_selection(
        transcript.id,
        root=store_root,
        state_root=state_root,
        candidate_id="",
        action="select",
        manual_candidate={"label": "Bob Buyer", "email": "bob@example.com"},
        reviewer="operator-test",
        note="Added from context workbench search.",
    )
    manual_state = manual_selection["context_workbench"]["contact_selection"]
    assert "Bob Buyer" in {candidate["label"] for candidate in manual_state["selected_candidates"]}
    search = transcript_api.search_context_contacts(
        transcript.id,
        root=store_root,
        state_root=state_root,
        query="bob",
    )
    assert search["items"][0]["email"] == "bob@example.com"
    instructions = transcript_api.record_context_instructions(
        transcript.id,
        root=store_root,
        state_root=state_root,
        instruction_text="Treat Bob as the purchasing contact and include the sample follow-up context.",
        reviewer="operator-test",
    )
    assert instructions["context_workbench"]["operator_context"]["status"] == "provided"
    preview = transcript_api.context_workbench_preview(transcript.id, root=store_root, state_root=state_root)
    manifest = json.loads(Path(preview["manifest"]).read_text(encoding="utf-8"))
    assert manifest["operator_context"]["instruction_text"].startswith("Treat Bob")
    assert any(candidate["email"] == "bob@example.com" for candidate in manifest["contact_selection"]["selected_candidates"])
    assert payload["final_preview"]["status"] == "blocked_identity_or_context_review"


def test_context_contact_candidates_merge_same_person_across_sources() -> None:
    candidates = transcript_api.unique_context_contact_candidates(
        [
            {
                "contact_id": "operator-sean",
                "label": "Sean Solberg",
                "source": "operator_participant_hint",
                "source_type": "operator_participant_hint",
                "source_profile": "user_config",
                "confidence": 0.9,
            },
            {
                "contact_id": "gws-sean",
                "label": "Solberg, Sean",
                "email": "ssolberg@fredlaw.com",
                "source": "gws_other_contact",
                "source_type": "gws_other_contact",
                "source_profile": "work",
                "confidence": 0.78,
            },
        ]
    )

    assert len(candidates) == 1
    assert candidates[0]["label"] == "Sean Solberg"
    assert candidates[0]["email"] == "ssolberg@fredlaw.com"
    assert candidates[0]["source_count"] == 2
    assert set(candidates[0]["merged_contact_ids"]) == {"operator-sean", "gws-sean"}


def test_context_contact_selection_batch_records_multiple_actions(tmp_path: Path, monkeypatch) -> None:
    store_root = tmp_path / "store"
    state_root = tmp_path / "state"
    state_root.mkdir()
    (state_root / participant_identity.CONTACT_SOURCE_CONFIG_NAME).write_text(
        json.dumps({"gws": {"profiles": [{"label": "work", "surfaces": ["contacts"]}]}}),
        encoding="utf-8",
    )
    transcript_path = write_transcript_artifact(tmp_path)
    add_calendar_attendee_context(transcript_path)

    def fake_gws(query_terms, *, config):
        return [
            ProvenanceSource(
                source_type="gws_contact",
                source_id="people/alice",
                label="Alice Example",
                snippet="Alice Example; alice@example.com",
                metadata={"profile": "work", "email": "alice@example.com"},
            )
        ]

    monkeypatch.setattr(participant_identity, "collect_gws_contact_provenance", fake_gws)
    transcript = transcript_store.ingest_artifact(
        transcript_path,
        root=store_root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
    )
    detail = transcript_api.get_conversation_detail(transcript.id, root=store_root, state_root=state_root)
    candidate_id = detail["identity_review"]["identity_bundle"]["contact_candidates"][0]["contact_id"]

    batch = transcript_api.record_context_contact_selection_batch(
        transcript.id,
        root=store_root,
        state_root=state_root,
        actions=[
            {"candidate_id": candidate_id, "action": "select", "reviewer": "operator-test"},
            {
                "candidate_id": "",
                "action": "select",
                "reviewer": "operator-test",
                "manual_candidate": {"label": "Bob Buyer", "email": "bob@example.com"},
            },
        ],
    )

    selection = batch["context_workbench"]["contact_selection"]
    assert batch["schema_version"] == "transcribe-audio.context-contact-selection-batch.v1"
    assert len(batch["decisions"]) == 2
    assert {candidate["email"] for candidate in selection["selected_candidates"]} == {
        "alice@example.com",
        "bob@example.com",
    }


def test_context_contact_search_refresh_caches_configured_source_results(tmp_path: Path, monkeypatch) -> None:
    store_root = tmp_path / "store"
    state_root = tmp_path / "state"
    state_root.mkdir()
    transcript_path = write_transcript_artifact(tmp_path)
    add_calendar_attendee_context(transcript_path)
    calls = []

    def fake_collect(query_terms, *, transcript, state_root):
        calls.append(list(query_terms))
        if query_terms == ["chris"]:
            return (
                [
                    ProvenanceSource(
                        source_type="gws_contact",
                        source_id="people/chris",
                        label="Chris Example",
                        snippet="Chris Example; chris@example.com",
                        metadata={"profile": "work", "email": "chris@example.com"},
                    )
                ],
                [{"source": "gws", "profile": "work", "read_only": True}],
                [],
            )
        return [], [], []

    monkeypatch.setattr(participant_identity, "collect_configured_contact_sources", fake_collect)
    transcript = transcript_store.ingest_artifact(
        transcript_path,
        root=store_root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
    )

    empty = transcript_api.search_context_contacts(
        transcript.id,
        root=store_root,
        state_root=state_root,
        query="chris",
    )
    calls_before_refresh = len(calls)
    refreshed = transcript_api.search_context_contacts(
        transcript.id,
        root=store_root,
        state_root=state_root,
        query="chris",
        mode="refresh",
    )
    cached = transcript_api.search_context_contacts(
        transcript.id,
        root=store_root,
        state_root=state_root,
        query="chris",
    )
    selected = transcript_api.record_context_contact_selection_batch(
        transcript.id,
        root=store_root,
        state_root=state_root,
        actions=[{"candidate_id": cached["items"][0]["contact_id"], "action": "select"}],
    )

    assert empty["total"] == 0
    assert refreshed["will_execute_external_action"] is True
    assert refreshed["items"][0]["label"] == "Chris Example"
    assert refreshed["cache_status"] == "updated"
    assert cached["items"][0]["email"] == "chris@example.com"
    assert calls[calls_before_refresh:] == [["chris"]]
    assert selected["context_workbench"]["contact_selection"]["selected_candidates"][0]["label"] == "Chris Example"


def test_context_contact_search_accepts_conversation_source_path(tmp_path: Path, monkeypatch) -> None:
    store_root = tmp_path / "store"
    state_root = tmp_path / "state"
    state_root.mkdir()
    transcript_path = write_transcript_artifact(tmp_path)
    add_calendar_attendee_context(transcript_path)

    def fake_collect(query_terms, *, transcript, state_root):
        if query_terms != ["hagberg"]:
            return [], [], []
        return (
            [
                ProvenanceSource(
                    source_type="gws_contact",
                    source_id="people/hagberg",
                    label="Erik C. Hagberg",
                    snippet="Erik C. Hagberg; erik.hagberg@example.com",
                    metadata={"profile": "work", "email": "erik.hagberg@example.com"},
                )
            ],
            [{"source": "gws", "profile": "work", "read_only": True}],
            [],
        )

    monkeypatch.setattr(participant_identity, "collect_configured_contact_sources", fake_collect)
    transcript = transcript_store.ingest_artifact(
        transcript_path,
        root=store_root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
    )

    refreshed = transcript_api.search_context_contacts(
        transcript.source_path,
        root=store_root,
        state_root=state_root,
        query="hagberg",
        mode="refresh",
    )
    cached = transcript_api.search_context_contacts(
        transcript.source_path,
        root=store_root,
        state_root=state_root,
        query="hagberg",
    )

    assert refreshed["items"][0]["label"] == "Erik C. Hagberg"
    assert cached["items"][0]["email"] == "erik.hagberg@example.com"


def test_context_contact_affinity_refresh_ranks_recent_frequent_contacts(tmp_path: Path) -> None:
    store_root = tmp_path / "store"
    state_root = tmp_path / "state"
    transcript = transcript_store.ingest_artifact(
        write_transcript_artifact(tmp_path),
        root=store_root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
    )
    detail = transcript_api.get_conversation_detail(transcript.id, root=store_root, state_root=state_root)
    conversation_key = detail["conversation"]["key"]
    recent = {
        "contact_id": "contact-chris-recent",
        "label": "Chris Recent",
        "email": "recent@example.com",
        "source": "gws_contact",
        "source_type": "gws_contact",
        "source_profile": "work",
        "confidence": 0.7,
    }
    stale = {
        "contact_id": "contact-chris-stale",
        "label": "Chris Stale",
        "email": "stale@example.com",
        "source": "gws_contact",
        "source_type": "gws_contact",
        "source_profile": "work",
        "confidence": 0.7,
    }
    transcript_api.append_context_contact_search_cache(
        state_root=state_root,
        conversation_key=conversation_key,
        query="chris",
        items=[stale, recent],
        source_profiles=[{"source": "gws", "profile": "work", "read_only": True}],
        warnings=[],
    )
    now = datetime.now(timezone.utc).replace(microsecond=0)
    selection_path = transcript_api.context_contact_selection_path(
        state_root=state_root,
        conversation_key=conversation_key,
    )
    transcript_api.write_json_file(
        selection_path,
        {
            "schema_version": "transcribe-audio.context-contact-selection.v1",
            "conversation_key": conversation_key,
            "decisions": [
                {
                    "candidate_id": recent["contact_id"],
                    "action": "select",
                    "created_at": (now - timedelta(days=3)).isoformat().replace("+00:00", "Z"),
                    "candidate": recent,
                },
                {
                    "candidate_id": recent["contact_id"],
                    "action": "select",
                    "created_at": (now - timedelta(days=20)).isoformat().replace("+00:00", "Z"),
                    "candidate": recent,
                },
                {
                    "candidate_id": stale["contact_id"],
                    "action": "exclude",
                    "created_at": (now - timedelta(days=320)).isoformat().replace("+00:00", "Z"),
                    "candidate": stale,
                },
            ],
        },
    )

    affinity = transcript_api.refresh_context_contact_affinity(
        transcript.id,
        root=store_root,
        state_root=state_root,
        query="chris",
    )
    cached_search = transcript_api.search_context_contacts(
        transcript.id,
        root=store_root,
        state_root=state_root,
        query="chris",
    )

    assert affinity["items"][0]["label"] == "Chris Recent"
    assert affinity["items"][0]["relationship_affinity"]["prior_selected_count"] == 2
    assert "selected before" in affinity["items"][0]["ranking_reasons"]
    assert cached_search["items"][0]["label"] == "Chris Recent"
    assert cached_search["items"][0]["rank_score"] > cached_search["items"][1]["rank_score"]


def test_context_contact_merge_batch_persists_reviewed_split(tmp_path: Path) -> None:
    store_root = tmp_path / "store"
    state_root = tmp_path / "state"
    transcript = transcript_store.ingest_artifact(
        write_transcript_artifact(tmp_path),
        root=store_root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
    )
    detail = transcript_api.get_conversation_detail(transcript.id, root=store_root, state_root=state_root)
    conversation_key = detail["conversation"]["key"]
    transcript_api.append_context_contact_search_cache(
        state_root=state_root,
        conversation_key=conversation_key,
        query="sean",
        items=[
            {
                "contact_id": "operator-sean",
                "label": "Sean Solberg",
                "source": "operator_participant_hint",
                "source_type": "operator_participant_hint",
                "source_profile": "user_config",
                "confidence": 0.9,
            },
            {
                "contact_id": "gws-sean",
                "label": "Solberg, Sean",
                "email": "ssolberg@fredlaw.com",
                "source": "gws_other_contact",
                "source_type": "gws_other_contact",
                "source_profile": "work",
                "confidence": 0.78,
            },
        ],
        source_profiles=[],
        warnings=[],
    )
    before = transcript_api.get_conversation_detail(transcript.id, root=store_root, state_root=state_root)
    before_candidates = before["context_workbench"]["contact_selection"]["searchable_candidates"]
    assert len([candidate for candidate in before_candidates if "sean" in candidate["label"].lower()]) == 1

    split = transcript_api.record_context_contact_merge_batch(
        transcript.id,
        root=store_root,
        state_root=state_root,
        actions=[
            {
                "action": "split",
                "contact_ids": ["operator-sean", "gws-sean"],
                "reviewer": "operator-test",
                "note": "Reviewed split test.",
            }
        ],
    )
    after_candidates = split["context_workbench"]["contact_selection"]["searchable_candidates"]
    sean_candidates = [candidate for candidate in after_candidates if "sean" in candidate["label"].lower()]

    assert split["schema_version"] == "transcribe-audio.context-contact-merge-batch.v1"
    assert split["context_workbench"]["contact_selection"]["merge_state"]["status"] == "reviewed"
    assert len(sean_candidates) == 2
    assert {candidate["contact_id"] for candidate in sean_candidates} == {"operator-sean", "gws-sean"}


def test_context_contact_refresh_writes_job_manifest(tmp_path: Path, monkeypatch) -> None:
    store_root = tmp_path / "store"
    state_root = tmp_path / "state"
    transcript_path = write_transcript_artifact(tmp_path)
    add_calendar_attendee_context(transcript_path)

    def fake_collect(query_terms, *, transcript, state_root):
        return (
            [
                ProvenanceSource(
                    source_type="gws_contact",
                    source_id="people/chris",
                    label="Chris Example",
                    snippet="Chris Example; chris@example.com",
                    metadata={"profile": "work", "email": "chris@example.com"},
                )
            ],
            [{"source": "gws", "profile": "work", "read_only": True}],
            [],
        )

    monkeypatch.setattr(participant_identity, "collect_configured_contact_sources", fake_collect)
    transcript = transcript_store.ingest_artifact(
        transcript_path,
        root=store_root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
    )

    refresh = transcript_api.refresh_context_contacts(
        transcript.id,
        root=store_root,
        state_root=state_root,
        query="chris",
    )
    job = transcript_api.read_context_contact_refresh_job(
        state_root=state_root,
        job_id=refresh["job_id"],
    )

    assert refresh["status"] == "completed"
    assert Path(refresh["job_path"]).exists()
    assert refresh["items"][0]["email"] == "chris@example.com"
    assert job["search"]["items"][0]["label"] == "Chris Example"
    assert job["will_perform_external_write"] is False


def test_conversation_identity_bundle_cache_avoids_repeated_contact_provenance(tmp_path: Path, monkeypatch) -> None:
    store_root = tmp_path / "store"
    state_root = tmp_path / "state"
    state_root.mkdir()
    (state_root / participant_identity.CONTACT_SOURCE_CONFIG_NAME).write_text(
        json.dumps({"gws": {"profiles": [{"label": "work", "surfaces": ["contacts"]}]}}),
        encoding="utf-8",
    )
    transcript_path = write_transcript_artifact(tmp_path)
    add_calendar_attendee_context(transcript_path)
    calls = {"gws": 0}

    def fake_gws(query_terms, *, config):
        calls["gws"] += 1
        return [
            ProvenanceSource(
                source_type="gws_contact",
                source_id="people/alice",
                label="Alice Example",
                snippet="Alice Example; alice@example.com",
                metadata={"profile": "work", "email": "alice@example.com"},
            )
        ]

    monkeypatch.setattr(participant_identity, "collect_gws_contact_provenance", fake_gws)
    transcript = transcript_store.ingest_artifact(
        transcript_path,
        root=store_root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
    )

    first = transcript_api.get_conversation_detail(transcript.id, root=store_root, state_root=state_root)
    second = transcript_api.get_conversation_detail(transcript.id, root=store_root, state_root=state_root)
    selection = transcript_api.record_context_contact_selection(
        transcript.id,
        root=store_root,
        state_root=state_root,
        candidate_id=first["identity_review"]["identity_bundle"]["contact_candidates"][0]["contact_id"],
        action="select",
    )

    assert calls["gws"] == 1
    assert first["identity_review"]["identity_cache"]["status"] == "stored"
    assert second["identity_review"]["identity_cache"]["status"] == "hit"
    assert selection["context_workbench"]["contact_selection"]["selected_candidates"][0]["label"] == "Alice Example"


def test_selected_first_pass_summary_prepare_is_conversation_scoped(tmp_path: Path) -> None:
    store_root = tmp_path / "store"
    state_root = tmp_path / "state"
    env_file = tmp_path / "auracall.env"
    env_file.write_text(
        "\n".join(
            [
                "OPENAI_BASE_URL=http://127.0.0.1:18095/v1",
                "OPENAI_API_KEY=test-key",
                "AURACALL_BATCH_URL=http://127.0.0.1:18095/v1/response-batches",
                "AURACALL_DISPATCH_TEAM=transcribe-audio-chatgpt-pro-pool",
                "AURACALL_DISPATCH_MODEL=gpt-5.2-pro",
            ]
        ),
        encoding="utf-8",
    )
    transcript = transcript_store.ingest_artifact(
        write_transcript_artifact(tmp_path),
        root=store_root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
    )

    prepared = transcript_api.prepare_selected_first_pass_summary(
        transcript.id,
        state_root=state_root,
        store_root=store_root,
        env_file=env_file,
        store=True,
    )
    status = transcript_api.selected_first_pass_summary_status(
        transcript.id,
        state_root=state_root,
        store_root=store_root,
        env_file=env_file,
        manifest=prepared["manifest"],
        materialize=True,
    )
    manifest = json.loads(Path(prepared["manifest"]).read_text(encoding="utf-8"))
    request = manifest["batch_payload"]["requests"][0]

    assert prepared["action"] == "prepare_selected_first_pass_summary"
    assert prepared["request_count"] == 1
    assert prepared["first_pass_summary"]["status"] == "needs_summary"
    assert prepared["will_execute_external_action"] is False
    assert manifest["queue"]["items"][0]["id"] == transcript.id
    assert manifest["batch_payload"]["metadata"]["scopedDocumentId"] == transcript.id
    assert request["metadata"]["transcriptDocumentId"] == transcript.id
    assert manifest["queue"]["items"][0]["participant_identity_bundle"]["schema_version"]
    assert "participant_identity" in request["input"][1]["content"]
    assert status["status"] == "prepared"


def test_selected_speaker_preprocessing_prepares_both_reviewed_phases(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store_root = tmp_path / "store"
    state_root = tmp_path / "state"
    transcript = transcript_store.ingest_artifact(
        write_transcript_artifact(tmp_path),
        root=store_root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
    )
    monkeypatch.setattr(
        transcript_api.provenance_config,
        "speaker_preprocessing_source_configs_from_provenance",
        lambda **kwargs: {
            "gws": [],
            "odollo": [],
            "source_contexts": [],
            "warnings": ["No eligible Source Context in test."],
        },
    )
    monkeypatch.setattr(
        transcript_api.speaker_identity_preprocess,
        "collect_configured_identity_evidence",
        lambda **kwargs: {
            "person_records": [],
            "provenance_sources": [],
            "source_contexts": [],
            "warnings": ["No bounded source results in test."],
        },
    )

    discovery = transcript_api.prepare_selected_speaker_clue_discovery(
        transcript.id,
        state_root=state_root,
        store_root=store_root,
    )
    evaluation = transcript_api.prepare_selected_speaker_identity_evaluation(
        transcript.id,
        state_root=state_root,
        store_root=store_root,
        evidence_mode="legacy_rollback",
        legacy_approval_token=(
            transcript_api.conversation_identity_policy
            .LEGACY_ROLLBACK_APPROVAL_TOKEN
        ),
        operator="test-operator",
        discovery_readout={
            "schema_version": "transcribe-audio.speaker-clue-discovery-readout.v1",
            "speaker_clues": [],
            "conversation_clues": [],
            "warnings": [],
        },
    )

    assert discovery["phase"] == "clue_discovery"
    assert discovery["will_send_prompt"] is False
    assert discovery["source_warnings"] == ["No eligible Source Context in test."]
    assert evaluation["phase"] == "identity_evaluation"
    assert evaluation["will_send_prompt"] is False
    assert evaluation["source_warnings"] == [
        (
            transcript_api.conversation_identity_policy
            .LEGACY_ROLLBACK_WARNING
        ),
        "No bounded source results in test.",
    ]
    assert Path(
        evaluation["legacy_rollback_receipt"]["receipt_path"]
    ).stat().st_mode & 0o777 == 0o600

    captured = transcript_api.persist_selected_speaker_identity_evaluation(
        transcript.id,
        state_root=state_root,
        store_root=store_root,
        identity_evaluation_run_id=evaluation["run_id"],
        clue_discovery_run_id=discovery["run_id"],
        readout={
            "schema_version": "transcribe-audio.speaker-identity-evaluation-readout.v1",
            "evaluation_id": evaluation["packet"]["evaluation_id"],
            "calendar_association": {"status": "ambiguous", "factors": []},
            "person_links": [],
            "speaker_assignments": [
                {
                    "speaker_labels": ["Speaker A"],
                    "status": "unresolved",
                    "person_id": "",
                    "factors": [],
                    "utterance_assignments": [],
                    "review_flags": ["insufficient_evidence"],
                }
            ],
            "warnings": [],
        },
    )
    state = transcript_api.selected_speaker_preprocessing_state(
        transcript.id,
        state_root=state_root,
        store_root=store_root,
    )
    proposal = state["current_evaluation"]["proposals"][0]
    decision = transcript_api.review_selected_speaker_proposal(
        transcript.id,
        state_root=state_root,
        store_root=store_root,
        evaluation_id=state["current_evaluation_id"],
        proposal_id=proposal["proposal_id"],
        action="defer",
        reviewer="test-operator",
        note="Spurious or insufficiently supported recording.",
    )

    assert captured["status"] == "awaiting_review"
    assert state["status"] == "awaiting_review"
    assert decision["record"]["review_decisions"][0]["action"] == "defer"


def test_selected_identity_evaluation_defaults_to_immutable_retrieval_bundle(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store_root = tmp_path / "store"
    state_root = tmp_path / "state"
    transcript = transcript_store.ingest_artifact(
        write_transcript_artifact(tmp_path),
        root=store_root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
    )
    source_path = Path(transcript.source_path)
    stored_path = Path(transcript.stored_path)
    source_bytes = source_path.read_bytes()
    stored_bytes = stored_path.read_bytes()
    with transcript_store.connect(store_root) as con:
        index_row_before = dict(
            con.execute(
                "SELECT * FROM documents WHERE id = ?",
                (transcript.id,),
            ).fetchone()
        )
    resolved = {
        "gws": [],
        "odollo": [],
        "source_contexts": [{"source_id": "gws-default"}],
        "retrieval_sources": [],
        "warnings": ["config warning"],
    }
    monkeypatch.setattr(
        transcript_api.provenance_config,
        "speaker_preprocessing_source_configs_from_provenance",
        lambda **kwargs: resolved,
    )
    bundle = SimpleNamespace(
        request=SimpleNamespace(
            request_id="00000000-0000-4000-8000-000000000801",
            budgets={"query_terms": ["person@example.com", "orchard"]},
        ),
        persisted_bundle=SimpleNamespace(
            bundle_id="00000000-0000-4000-8000-000000000802",
            content_hash="bundle-hash",
            status="partial",
        ),
        source_failures=(
            {
                "adapter_id": "gws-evidence-v1",
                "reason_code": "provider_unavailable",
            },
        ),
        warnings=("no_bounded_evidence",),
    )
    retrieved = SimpleNamespace(
        bundle=bundle,
        policy_build=SimpleNamespace(
            source_contexts=tuple(resolved["source_contexts"]),
            warnings=tuple(resolved["warnings"]),
        ),
        projection_receipt=SimpleNamespace(
            receipt_path="/private/projection.json"
        ),
        shadow_root=Path("/private/shadow"),
        retrieval_receipt_path=Path("/private/retrieval.json"),
        retrieval_receipt_sha256="retrieval-receipt-hash",
        preparation_transcript_path=state_root / "private-snapshot.transcript.json",
        source_transcript_sha256="source-hash",
        preparation_transcript_sha256="snapshot-hash",
        source_was_derived=True,
    )
    retrieval_calls = []
    monkeypatch.setattr(
        transcript_api.conversation_identity_policy,
        "prepare_transcript_identity_evidence",
        lambda *args, **kwargs: (
            retrieval_calls.append((args, kwargs)) or retrieved
        ),
    )
    monkeypatch.setattr(
        transcript_api.speaker_identity_preprocess,
        "collect_configured_identity_evidence",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("default retrieval must not call legacy evidence")
        ),
    )
    prepared_calls = []
    monkeypatch.setattr(
        transcript_api.speaker_preprocessing_workflow,
        "prepare_identity_evaluation",
        lambda *args, **kwargs: (
            prepared_calls.append((args, kwargs))
            or {
                "phase": "identity_evaluation",
                "will_send_prompt": False,
                "packet": {},
            }
        ),
    )

    result = transcript_api.prepare_selected_speaker_identity_evaluation(
        transcript.id,
        state_root=state_root,
        store_root=store_root,
        discovery_readout={
            "schema_version": (
                "transcribe-audio.speaker-clue-discovery-readout.v1"
            ),
            "speaker_clues": [],
            "conversation_clues": [],
            "warnings": [],
        },
    )

    assert len(retrieval_calls) == 1
    assert retrieval_calls[0][1]["resolved"] is resolved
    assert retrieval_calls[0][1]["source_store_root"] == store_root
    assert retrieval_calls[0][1]["document_id"] == transcript.id
    assert prepared_calls[0][0][0] == retrieved.preparation_transcript_path
    assert prepared_calls[0][1]["retrieval_bundle"] is bundle
    assert result["evidence_mode"] == "retrieval"
    assert result["legacy_rollback_receipt"] == {}
    assert result["retrieval"]["status"] == "partial"
    assert result["retrieval"]["query_terms"] == [
        "person@example.com",
        "orchard",
    ]
    assert result["retrieval"]["retrieval_receipt_sha256"] == (
        "retrieval-receipt-hash"
    )
    assert result["source_warnings"] == [
        "config warning",
        "no_bounded_evidence",
    ]
    assert result["retrieval"]["source_transcript_sha256"] == "source-hash"
    assert result["retrieval"]["preparation_transcript_sha256"] == "snapshot-hash"
    assert result["retrieval"]["source_was_derived"] is True
    assert source_path.read_bytes() == source_bytes
    assert stored_path.read_bytes() == stored_bytes
    with transcript_store.connect(store_root) as con:
        index_row_after = dict(
            con.execute(
                "SELECT * FROM documents WHERE id = ?",
                (transcript.id,),
            ).fetchone()
        )
    assert index_row_after == index_row_before


def test_selected_speaker_reference_repair_uses_original_ledger_packet(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store_root = tmp_path / "store"
    state_root = tmp_path / "state"
    transcript = transcript_store.ingest_artifact(
        write_transcript_artifact(tmp_path),
        root=store_root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
    )
    monkeypatch.setattr(
        transcript_api.provenance_config,
        "speaker_preprocessing_source_configs_from_provenance",
        lambda **kwargs: {
            "gws": [],
            "odollo": [],
            "source_contexts": [],
            "warnings": [],
        },
    )
    discovery = transcript_api.prepare_selected_speaker_clue_discovery(
        transcript.id,
        state_root=state_root,
        store_root=store_root,
    )
    rejected_readout = {
        "schema_version": "transcribe-audio.speaker-clue-discovery-readout.v1",
        "speaker_clues": [
            {
                "speaker_label": "Speaker A",
                "transcript_clue_ids": ["utterance-99"],
                "observations": [],
                "person_hints": [],
                "retrieval_terms": [],
            }
        ],
        "conversation_clues": [],
        "warnings": [],
    }
    monkeypatch.setattr(
        transcript_api.speaker_preprocessing_workflow,
        "captured_run_json",
        lambda **kwargs: rejected_readout,
    )
    original_packet_path = Path(discovery["input_packet_path"])
    original_packet_bytes = original_packet_path.read_bytes()

    repair = transcript_api.prepare_selected_speaker_reference_repair(
        transcript.id,
        state_root=state_root,
        store_root=store_root,
        phase="clue_discovery",
        original_run_id=discovery["run_id"],
        route=discovery["route"],
    )

    assert repair["phase"] == "clue_discovery_reference_repair"
    assert repair["repair_packet"]["original_run_id"] == discovery["run_id"]
    assert repair["repair_packet"]["rejected_readout"] == rejected_readout
    assert original_packet_path.read_bytes() == original_packet_bytes
    assert Path(repair["input_packet_path"]) != original_packet_path
    assert repair["will_perform_external_write"] is False


def test_selected_speaker_preprocessing_prepares_verified_stored_fallback_and_syncs_ids(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store_root = tmp_path / "store"
    state_root = tmp_path / "state"
    source_path = write_transcript_artifact(tmp_path)
    transcript = transcript_store.ingest_artifact(
        source_path,
        root=store_root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
    )
    source_path.unlink()
    monkeypatch.setattr(
        transcript_api.provenance_config,
        "speaker_preprocessing_source_configs_from_provenance",
        lambda **kwargs: {
            "gws": [],
            "odollo": [],
            "source_contexts": [],
            "warnings": [],
        },
    )

    discovery = transcript_api.prepare_selected_speaker_clue_discovery(
        transcript.id,
        state_root=state_root,
        store_root=store_root,
    )

    stored_path = Path(transcript.stored_path)
    stored_payload = json.loads(stored_path.read_text(encoding="utf-8"))
    stored_document = transcript_api.get_document(transcript.id, root=store_root)
    assert discovery["phase"] == "clue_discovery"
    assert discovery["transcript_artifact"]["location"] == "stored"
    assert stored_payload["conversation_id"]
    assert stored_payload["recording_id"]
    assert (
        stored_document["json_payload"]["conversation_id"]
        == stored_payload["conversation_id"]
    )
    assert (
        stored_document["json_payload"]["recording_id"]
        == stored_payload["recording_id"]
    )
    assert stored_document["artifact_sha256"] == transcript_store.sha256_file(
        stored_path
    )


def test_speaker_evaluation_campaign_api_applies_and_opens_private_review_packet(
    tmp_path: Path,
) -> None:
    store_root = tmp_path / "store"
    state_root = tmp_path / "state"
    transcript = transcript_store.ingest_artifact(
        write_transcript_artifact(tmp_path),
        root=store_root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
    )
    server = transcript_api.TranscriptApiServer(
        ("127.0.0.1", 0),
        transcript_api.TranscriptApiHandler,
        store_root=store_root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
        state_root=state_root,
        quiet=True,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        request = Request(
            f"http://{host}:{port}/api/speaker-evaluation-campaigns/apply",
            data=json.dumps(
                {
                    "batch_size": 1,
                    "approval_token": (
                        "APPLY_SPEAKER_EVALUATION_CAMPAIGN_MANIFEST"
                    ),
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        applied_response = urlopen(request, timeout=5)
        applied = json.loads(applied_response.read())
        status = json.loads(
            urlopen(
                f"http://{host}:{port}/api/speaker-evaluation-campaigns/"
                f"{quote(applied['campaign_id'])}",
                timeout=5,
            ).read()
        )
        packet = json.loads(
            urlopen(
                f"http://{host}:{port}/api/speaker-evaluation-campaigns/"
                f"{quote(applied['campaign_id'])}/cases/{quote(transcript.id)}/"
                "review-packet",
                timeout=5,
            ).read()
        )
    finally:
        server.shutdown()
        server.server_close()

    assert applied_response.status == 201
    assert status["reviewed_case_count"] == 0
    assert status["eligible_known_count"] == 0
    assert packet["document_id"] == transcript.id
    assert packet["will_read_gold_records"] is False
    assert packet["will_execute_app_intelligence"] is False


def test_selected_first_pass_summary_run_endpoint_prepares_and_submits(tmp_path: Path) -> None:
    store_root = tmp_path / "store"
    state_root = tmp_path / "state"
    env_file = tmp_path / "auracall.env"
    transcript = transcript_store.ingest_artifact(
        write_transcript_artifact(tmp_path),
        root=store_root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
    )
    FakeAuraCallHandler.requests = []
    provider = ThreadingHTTPServer(("127.0.0.1", 0), FakeAuraCallHandler)
    provider_thread = threading.Thread(target=provider.serve_forever, daemon=True)
    provider_thread.start()
    host, provider_port = provider.server_address
    env_file.write_text(
        "\n".join(
            [
                f"OPENAI_BASE_URL=http://{host}:{provider_port}/v1",
                "OPENAI_API_KEY=test-key",
                f"AURACALL_BATCH_URL=http://{host}:{provider_port}/v1/response-batches",
                "AURACALL_DISPATCH_TEAM=transcribe-audio-chatgpt-pro-pool",
                "AURACALL_DISPATCH_MODEL=gpt-5.2-pro",
            ]
        ),
        encoding="utf-8",
    )
    server = transcript_api.TranscriptApiServer(
        ("127.0.0.1", 0),
        transcript_api.TranscriptApiHandler,
        store_root=store_root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
        state_root=state_root,
        batch_env_file=env_file,
        quiet=True,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        api_host, port = server.server_address
        blocked_request = Request(
            f"http://{api_host}:{port}/api/conversations/{quote(transcript.id)}/first-pass-summary/run",
            data=json.dumps({"store": True}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urlopen(blocked_request, timeout=5)
        except HTTPError as exc:
            assert exc.code == 400
            assert "approval_token" in json.loads(exc.read())["error"]
        else:
            raise AssertionError("one-click run without approval token must fail")
        assert FakeAuraCallHandler.requests == []

        run_request = Request(
            f"http://{api_host}:{port}/api/conversations/{quote(transcript.id)}/first-pass-summary/run",
            data=json.dumps(
                {
                    "store": True,
                    "approval_token": "SUBMIT_FIRST_PASS_SUMMARY_BATCH",
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        run_response = urlopen(run_request, timeout=5)
        payload = json.loads(run_response.read())
        manifest = json.loads(Path(payload["manifest"]).read_text(encoding="utf-8"))
    finally:
        server.shutdown()
        server.server_close()
        provider.shutdown()
        provider.server_close()

    assert run_response.status == 202
    assert payload["action"] == "run_selected_first_pass_summary"
    assert payload["prepared"]["status"] == "prepared"
    assert payload["status"] == "submitted"
    assert payload["batch_id"] == "batch_test"
    assert payload["one_click"] is True
    assert payload["will_execute_external_action"] is True
    assert payload["will_perform_external_write"] is True
    assert manifest["dry_run"] is False
    assert manifest["batch"]["id"] == "batch_test"
    assert manifest["batch_payload"]["metadata"]["scopedDocumentId"] == transcript.id
    assert FakeAuraCallHandler.requests[0]["metadata"]["workflow"] == "transcribe-audio-first-pass-summary"


def test_speaker_identity_review_records_contact_and_defer_queue(tmp_path: Path) -> None:
    store_root = tmp_path / "store"
    state_root = tmp_path / "state"
    transcript_path = write_transcript_artifact(tmp_path)
    transcript = transcript_store.ingest_artifact(
        transcript_path,
        root=store_root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
    )

    confirmed = transcript_api.record_speaker_identity_review(
        transcript.id,
        root=store_root,
        state_root=state_root,
        speaker_label="Speaker A",
        action="confirm",
        contact_label="Alice Example",
        reviewer="api-test",
    )
    deferred = transcript_api.record_speaker_identity_review(
        transcript.id,
        root=store_root,
        state_root=state_root,
        speaker_label="Speaker B",
        action="defer",
        reviewer="api-test",
    )
    readout_delegated = transcript_api.record_speaker_identity_review(
        transcript.id,
        root=store_root,
        state_root=state_root,
        speaker_label="Speaker C",
        action="llm_readout",
        reviewer="api-test",
    )
    queue = transcript_api.review_queue_summary(state_root=state_root, store_root=store_root, limit=20)

    assert confirmed["status"] == "confirmed"
    assert confirmed["identity_review"]["confirmed_count"] == 1
    assert deferred["status"] == "deferred"
    assert readout_delegated["status"] == "llm_readout"
    speaker_bucket = next(bucket for bucket in queue["buckets"] if bucket["id"] == "speaker_ids")
    assert speaker_bucket["count"] == 1
    assert any(item["type"] == "speaker_identity_review" and item["workflow_stage"] == "speakers" for item in queue["items"])
    with transcript_api.connect(store_root) as con:
        row = con.execute(
            "SELECT status, evidence_json FROM speaker_assignments WHERE speaker_label = ?",
            ("Speaker C",),
        ).fetchone()
    assert row["status"] == "llm_readout"
    assert json.loads(row["evidence_json"])[0]["source"] == "operator_readout_delegation"


def test_context_and_final_preview_actions_write_local_review_records(tmp_path: Path) -> None:
    store_root = tmp_path / "store"
    state_root = tmp_path / "state"
    transcript_path = write_transcript_artifact(tmp_path)
    readout_path = write_readout_artifact(tmp_path, transcript_path)
    route_path = write_route_artifact(tmp_path, transcript_path, readout_path)
    transcript_store.ingest_artifact(
        transcript_path,
        root=store_root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
    )
    transcript_store.ingest_artifact(
        readout_path,
        root=store_root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
    )
    contextual = transcript_store.ingest_artifact(
        write_contextual_readout_artifact(tmp_path, transcript_path, route_path),
        root=store_root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
    )

    context_action = transcript_api.context_workbench_preview(
        contextual.id,
        root=store_root,
        state_root=state_root,
        queue=True,
        approval_token="QUEUE_CONTEXT_WORKBENCH_RUN",
    )
    preview_action = transcript_api.queue_deposition_memory_preview(
        contextual.id,
        root=store_root,
        state_root=state_root,
        approval_token="QUEUE_DEPOSITION_MEMORY_PREVIEW",
    )
    queue = transcript_api.review_queue_summary(state_root=state_root, store_root=store_root, limit=20)

    assert context_action["status"] == "queued"
    assert context_action["will_run_provider"] is False
    assert Path(context_action["manifest"]).exists()
    context_manifest = json.loads(Path(context_action["manifest"]).read_text(encoding="utf-8"))
    assert context_manifest["participant_identity_bundle"]["schema_version"]
    assert context_manifest["contact_selection"]["schema_version"]
    assert preview_action["status"] == "blocked_identity_or_context_review"
    assert preview_action["will_perform_external_write"] is False
    assert preview_action["final_preview"]["identity_context_warnings"]
    preview_bucket = next(bucket for bucket in queue["buckets"] if bucket["id"] == "deposition_memory_preview")
    assert preview_bucket["count"] == 0


def test_retranscription_preflight_resolves_readout_source_blob_without_work(tmp_path: Path) -> None:
    store_root = tmp_path / "store"
    state_root = tmp_path / "state"
    transcript_path = write_transcript_artifact(tmp_path)
    transcript = transcript_store.ingest_artifact(
        transcript_path,
        root=store_root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
    )
    readout = transcript_store.ingest_artifact(
        write_readout_artifact(tmp_path, transcript_path),
        root=store_root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
    )

    payload = transcript_api.retranscription_preflight(
        readout.id,
        root=store_root,
        state_root=state_root,
        backend="assemblyai",
    )

    assert payload["ok"] is True
    assert payload["selected_backend"] == "assemblyai"
    assert payload["source_document"]["id"] == transcript.id
    assert payload["source_blob"]["id"]
    assert payload["planned_outputs"]["output_dir"] == str(state_root / "retranscriptions" / transcript.id)
    assert payload["command"][1] == "assembly_transcribe.py"
    assert payload["will_queue"] is False
    assert payload["will_run_transcription"] is False
    assert payload["will_write_files"] is False
    assert payload["future_required_approval_token_for_queue"] == "QUEUE_RETRANSCRIPTION_JOB"


def test_retranscription_preflight_endpoint_is_dry_run_only(tmp_path: Path) -> None:
    store_root = tmp_path / "store"
    transcript_path = write_transcript_artifact(tmp_path)
    readout = transcript_store.ingest_artifact(
        write_readout_artifact(tmp_path, transcript_path),
        root=store_root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
    )
    transcript_store.ingest_artifact(
        transcript_path,
        root=store_root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
    )
    server = transcript_api.TranscriptApiServer(
        ("127.0.0.1", 0),
        transcript_api.TranscriptApiHandler,
        store_root=store_root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
        state_root=tmp_path / "state",
        quiet=True,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        response = urlopen(
            Request(
                f"http://{host}:{port}/api/documents/{quote(readout.id)}/retranscription/preflight",
                data=json.dumps({"backend": "faster_whisper"}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            ),
            timeout=5,
        )
        payload = json.loads(response.read())
    finally:
        server.shutdown()
        server.server_close()

    assert response.status == 200
    assert payload["schema_version"] == "transcribe-audio.retranscription-preflight.v1"
    assert payload["selected_backend"] == "faster_whisper"
    assert payload["will_queue"] is False
    assert payload["will_run_transcription"] is False
    assert payload["will_write_files"] is False


def test_enqueue_retranscription_job_writes_manifest_without_running_backend(tmp_path: Path) -> None:
    store_root = tmp_path / "store"
    state_root = tmp_path / "state"
    transcript_path = write_transcript_artifact(tmp_path)
    readout = transcript_store.ingest_artifact(
        write_readout_artifact(tmp_path, transcript_path),
        root=store_root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
    )
    transcript_store.ingest_artifact(
        transcript_path,
        root=store_root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
    )

    payload = transcript_api.enqueue_retranscription_job(
        readout.id,
        root=store_root,
        state_root=state_root,
        backend="assemblyai",
        approval_token="QUEUE_RETRANSCRIPTION_JOB",
    )

    job_path = Path(payload["job"]["path"])
    job_payload = json.loads(job_path.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["status"] == "queued"
    assert payload["required_approval_token_checked"] == "QUEUE_RETRANSCRIPTION_JOB"
    assert payload["will_start_background_job"] is False
    assert payload["will_run_transcription"] is False
    assert payload["will_write_files"] is False
    assert payload["future_required_approval_token_for_run"] == "RUN_RETRANSCRIPTION_JOB"
    assert job_path.parent == state_root / "retranscription-jobs"
    assert job_payload["status"] == "queued"
    assert job_payload["selected_backend"] == "assemblyai"
    assert job_payload["will_run_transcription"] is False
    assert job_payload["will_write_files"] is False


def test_retranscription_queue_endpoint_requires_approval_token(tmp_path: Path) -> None:
    store_root = tmp_path / "store"
    transcript_path = write_transcript_artifact(tmp_path)
    readout = transcript_store.ingest_artifact(
        write_readout_artifact(tmp_path, transcript_path),
        root=store_root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
    )
    transcript_store.ingest_artifact(
        transcript_path,
        root=store_root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
    )
    server = transcript_api.TranscriptApiServer(
        ("127.0.0.1", 0),
        transcript_api.TranscriptApiHandler,
        store_root=store_root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
        state_root=tmp_path / "state",
        quiet=True,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        try:
            urlopen(
                Request(
                    f"http://{host}:{port}/api/documents/{quote(readout.id)}/retranscription/queue",
                    data=json.dumps({"backend": "faster_whisper"}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                ),
                timeout=5,
            )
        except HTTPError as exc:
            assert exc.code == HTTPStatus.BAD_REQUEST
            assert "approval_token" in json.loads(exc.read())["error"]
        else:
            raise AssertionError("retranscription queue without approval token must fail")
        response = urlopen(
            Request(
                f"http://{host}:{port}/api/documents/{quote(readout.id)}/retranscription/queue",
                data=json.dumps(
                    {
                        "backend": "faster_whisper",
                        "approval_token": "QUEUE_RETRANSCRIPTION_JOB",
                    }
                ).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            ),
            timeout=5,
        )
        payload = json.loads(response.read())
    finally:
        server.shutdown()
        server.server_close()

    assert response.status == HTTPStatus.CREATED
    assert payload["ok"] is True
    assert payload["job"]["status"] == "queued"
    assert payload["will_start_background_job"] is False
    assert payload["will_run_transcription"] is False
    assert payload["will_write_files"] is False


def test_library_and_search_use_store_documents(tmp_path: Path) -> None:
    store_root = tmp_path / "store"
    transcript_store.ingest_artifact(
        write_transcript_artifact(tmp_path),
        root=store_root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
    )

    library = transcript_api.list_documents(root=store_root, limit=10)
    search = transcript_store.search_store(
        "Tempo Chemical",
        root=store_root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
    )

    assert library["total"] == 1
    assert library["items"][0]["media_blob"]["id"]
    assert search[0]["title"] == "Weekly Product Sync"


def test_blob_route_supports_range_reads(tmp_path: Path) -> None:
    store_root = tmp_path / "store"
    result = transcript_store.ingest_artifact(
        write_transcript_artifact(tmp_path),
        root=store_root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
    )
    blob_id = transcript_api.get_document(result.id, root=store_root)["blobs"][0]["id"]
    server = transcript_api.TranscriptApiServer(
        ("127.0.0.1", 0),
        transcript_api.TranscriptApiHandler,
        store_root=store_root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
        quiet=True,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        request = Request(f"http://{host}:{port}/api/blobs/{blob_id}", headers={"Range": "bytes=2-5"})
        response = urlopen(request, timeout=5)
        assert response.status == 206
        assert response.headers["Content-Range"] == "bytes 2-5/16"
        assert response.read() == b"2345"
    finally:
        server.shutdown()
        server.server_close()


def test_static_frontend_serves_index_and_assets(tmp_path: Path) -> None:
    store_root = tmp_path / "store"
    static_root = tmp_path / "static"
    static_root.mkdir()
    (static_root / "index.html").write_text('<div id="root"></div><script src="/assets/app.js"></script>', encoding="utf-8")
    assets = static_root / "assets"
    assets.mkdir()
    (assets / "app.js").write_text("console.log('ok')", encoding="utf-8")
    server = transcript_api.TranscriptApiServer(
        ("127.0.0.1", 0),
        transcript_api.TranscriptApiHandler,
        store_root=store_root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
        quiet=True,
        static_dir=static_root,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        index = urlopen(f"http://{host}:{port}/", timeout=5)
        asset = urlopen(f"http://{host}:{port}/assets/app.js", timeout=5)
        fallback = urlopen(f"http://{host}:{port}/library/deep-link", timeout=5)
        assert index.status == 200
        assert b'<div id="root">' in index.read()
        assert asset.headers["Content-Type"].startswith("text/javascript") or asset.headers["Content-Type"].startswith(
            "application/javascript"
        )
        assert b"console.log" in asset.read()
        assert b'<div id="root">' in fallback.read()
        try:
            urlopen(f"http://{host}:{port}/api/missing", timeout=5)
        except HTTPError as exc:
            assert exc.code == 404
            assert json.loads(exc.read())["error"] == "Not found"
        else:
            raise AssertionError("Unknown API routes must not fall through to the SPA")
    finally:
        server.shutdown()
        server.server_close()


def test_review_queue_summary_reads_local_state(tmp_path: Path) -> None:
    store_root = tmp_path / "store"
    state_root = tmp_path / "state"
    route_path = tmp_path / "meeting.route.json"
    route_path.write_text(
        json.dumps(
            {
                "selected_candidate": {
                    "label": "SoyLei Tempo Chemical matter",
                    "target_kind": "matter",
                    "confidence": 0.62,
                }
            }
        ),
        encoding="utf-8",
    )
    review_dir = state_root / "review-queue"
    review_dir.mkdir(parents=True)
    (review_dir / "route-a.route-review.json").write_text(
        json.dumps(
            {
                "created_at": "2026-05-16T21:00:00Z",
                "reason": "Route confidence below threshold.",
                "route_decision_path": str(route_path),
                "selected_label": "SoyLei Tempo Chemical matter",
            }
        ),
        encoding="utf-8",
    )
    (review_dir / "route-b.route-review.json").write_text(
        json.dumps(
            {
                "created_at": "2026-05-16T21:01:00Z",
                "reason": "Route confidence below threshold.",
                "route_decision_path": str(tmp_path / "missing.route.json"),
                "selected_label": "Missing route",
            }
        ),
        encoding="utf-8",
    )
    conflict_dir = state_root / "filename-conflict-reviews"
    conflict_dir.mkdir()
    (conflict_dir / "filename-conflict-review-20260516-153723.json").write_text(
        json.dumps(
            {
                "created_at": "2026-05-16T20:37:23Z",
                "items": [
                    {"id": "one", "decision": "keep_target"},
                    {"id": "two", "decision": "pending"},
                ],
            }
        ),
        encoding="utf-8",
    )
    app_intelligence_ledger.create_run(
        state_root=state_root,
        workflow="contextual_reread",
        purpose="Review app-server decision.",
        document_id="doc-abc",
        run_id="human-review-run",
    )
    run_payload = app_intelligence_ledger.response_for_run(state_root=state_root, run_id="human-review-run")["run"]
    decision_dir = state_root / "app-intelligence-runs" / "human-review-run" / "artifacts" / "structured-decisions"
    decision_dir.mkdir(parents=True)
    decision_path = decision_dir / "decision-1.json"
    decision_path.write_text(
        json.dumps(
            {
                "schema_version": "transcribe-audio.app-intelligence-structured-decision-validation.v1",
                "decision_id": "decision-1",
                "run_id": "human-review-run",
                "valid": True,
                "decision": {
                    "action": "ask_for_human_review",
                    "rationale": "Operator should review ambiguous context.",
                    "confidence": 0.7,
                    "review_flags": ["ambiguous_context"],
                },
            }
        ),
        encoding="utf-8",
    )
    run_payload["decisions"] = [
        {
            "decision_id": "decision-1",
            "valid": True,
            "action": "ask_for_human_review",
            "status": "validated",
            "artifact_path": str(decision_path),
            "will_execute_host_action": False,
            "created_at": "2026-05-16T21:02:00Z",
        }
    ]
    (state_root / "app-intelligence-runs" / "human-review-run" / "run.json").write_text(
        json.dumps(run_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    payload = transcript_api.review_queue_summary(state_root=state_root, store_root=store_root, limit=20)

    route_bucket = next(bucket for bucket in payload["buckets"] if bucket["id"] == "route_reviews")
    app_review_bucket = next(bucket for bucket in payload["buckets"] if bucket["id"] == "app_intelligence_human_review")
    conflict_bucket = next(bucket for bucket in payload["buckets"] if bucket["id"] == "filename_conflicts")
    summary_bucket = next(bucket for bucket in payload["buckets"] if bucket["id"] == "first_pass_summaries")
    assert route_bucket["count"] == 1
    assert route_bucket["stale_count"] == 1
    assert app_review_bucket["count"] == 1
    assert app_review_bucket["pending_apply_count"] == 1
    assert conflict_bucket["count"] == 1
    assert summary_bucket["label"] == "First-pass summaries"
    assert conflict_bucket["decisions"] == {"keep_target": 1, "pending": 1}
    assert {item["status"] for item in payload["items"]} == {"pending", "stale_reference", "pending_apply"}
    assert any(item["bucket"] == "app_intelligence_human_review" for item in payload["items"])


def test_codex_app_server_provider_readiness(monkeypatch) -> None:
    def fake_run(args, check=False, capture_output=True, text=True, timeout=10):
        assert check is False
        assert capture_output is True
        assert text is True
        assert timeout == 10
        stdout = "codex-cli 0.131.0\n"
        if args[1:] == ["app-server", "--help"]:
            stdout = "Usage: codex app-server [OPTIONS]\n      --listen <URL>\n      --ws-auth <MODE>\n"
        return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(transcript_api.shutil, "which", lambda name: "/usr/local/bin/codex")
    monkeypatch.setattr(transcript_api.subprocess, "run", fake_run)

    provider = transcript_api.codex_app_server_readiness()

    assert provider["id"] == "codex-app-server"
    assert provider["status"] == "ready"
    assert provider["ready"] is True
    assert provider["recommended_transport"] == "stdio"
    assert provider["capabilities"]["persistent_sessions"] is True
    assert provider["capabilities"]["remote_transport"] is True
    assert provider["capabilities"]["websocket_auth"] is True


def test_intelligence_providers_endpoint_includes_app_server(tmp_path: Path, monkeypatch) -> None:
    def fake_run(args, check=False, capture_output=True, text=True, timeout=10):
        stdout = "codex-cli 0.131.0\n"
        if args[1:] == ["app-server", "--help"]:
            stdout = "Usage: codex app-server [OPTIONS]\n      --listen <URL>\n      --ws-auth <MODE>\n"
        return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(transcript_api.shutil, "which", lambda name: "/usr/local/bin/codex")
    monkeypatch.setattr(transcript_api.subprocess, "run", fake_run)

    server = transcript_api.TranscriptApiServer(
        ("127.0.0.1", 0),
        transcript_api.TranscriptApiHandler,
        store_root=tmp_path / "store",
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
        state_root=tmp_path / "state",
        quiet=True,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        payload = json.loads(urlopen(f"http://{host}:{port}/api/intelligence/providers", timeout=5).read())
    finally:
        server.shutdown()
        server.server_close()

    providers = {provider["id"]: provider for provider in payload["providers"]}
    assert payload["default_supervisor"] == "codex-app-server"
    assert providers["codex-app-server"]["status"] == "ready"
    assert providers["codex-app-server"]["control_plane"] == "codex-app-server"


def test_intelligence_smokes_endpoint_reports_latest_evidence(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    run_dir = state_root / "app-intelligence-runs" / "smoke-replay-manifest-test"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": "smoke-replay-manifest-test",
                "workflow": "app_replay_manifest_smoke",
                "status": "running",
                "phase": "session_started",
                "updated_at": "2026-05-21T12:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    evidence_dir = state_root / "browser-smokes"
    evidence_dir.mkdir(parents=True)
    screenshot = evidence_dir / "smoke.png"
    screenshot.write_bytes(b"png")
    report = evidence_dir / "smoke.json"
    report.write_text(
        json.dumps(
            {
                "status": "pass",
                "run_id": "smoke-replay-manifest-test",
                "screenshot_path": str(screenshot),
                "missing_checks": [],
                "checks": {"hasReplayManifest": True, "hasNoWriteFlag": True},
            }
        ),
        encoding="utf-8",
    )
    server = transcript_api.TranscriptApiServer(
        ("127.0.0.1", 0),
        transcript_api.TranscriptApiHandler,
        store_root=tmp_path / "store",
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
        state_root=state_root,
        quiet=True,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        payload = json.loads(urlopen(f"http://{host}:{port}/api/intelligence/smokes", timeout=5).read())
    finally:
        server.shutdown()
        server.server_close()

    assert payload["schema_version"] == "transcribe-audio.app-smoke-status.v1"
    assert payload["will_read_artifact_content"] is False
    assert payload["will_execute_write_bearing_action"] is False
    assert payload["latest_report"]["status"] == "pass"
    assert payload["latest_report"]["screenshot_exists"] is True
    assert payload["latest_report"]["checks"]["hasReplayManifest"] is True
    assert payload["runs"][0]["run_id"] == "smoke-replay-manifest-test"


def test_intelligence_smoke_jobs_endpoint_queues_allowlisted_command(tmp_path: Path, monkeypatch) -> None:
    def fake_run(args, cwd=None, check=False, capture_output=True, text=True, timeout=180):
        assert "smoke_app_replay_manifest.py" in args[1]
        assert "--state-root" in args
        assert check is False
        assert capture_output is True
        assert text is True
        return subprocess.CompletedProcess(args, 0, stdout='APP_REPLAY_MANIFEST_SMOKE_JSON={"status":"pass"}\n', stderr="")

    monkeypatch.setattr(transcript_api.subprocess, "run", fake_run)
    state_root = tmp_path / "state"
    server = transcript_api.TranscriptApiServer(
        ("127.0.0.1", 0),
        transcript_api.TranscriptApiHandler,
        store_root=tmp_path / "store",
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
        state_root=state_root,
        quiet=True,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        try:
            urlopen(
                Request(
                    f"http://{host}:{port}/api/intelligence/smoke-jobs",
                    data=json.dumps({"job_type": "api_replay_smoke"}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                ),
                timeout=5,
            )
        except HTTPError as exc:
            assert exc.code == HTTPStatus.BAD_REQUEST
            assert "approval_token" in json.loads(exc.read())["error"]
        else:
            raise AssertionError("smoke job without approval token must fail")
        created = json.loads(
            urlopen(
                Request(
                    f"http://{host}:{port}/api/intelligence/smoke-jobs",
                    data=json.dumps(
                        {
                            "job_type": "api_replay_smoke",
                            "approval_token": "RUN_APP_SMOKE_JOB",
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                ),
                timeout=5,
            ).read()
        )
        job_id = created["job"]["job_id"]
        for _ in range(20):
            payload = json.loads(urlopen(f"http://{host}:{port}/api/intelligence/smoke-jobs", timeout=5).read())
            job = next(item for item in payload["items"] if item["job_id"] == job_id)
            if job["status"] == "succeeded":
                break
            time.sleep(0.05)
        else:
            raise AssertionError("smoke job did not complete")
    finally:
        server.shutdown()
        server.server_close()

    assert created["will_execute_arbitrary_shell"] is False
    assert job["will_read_artifact_content"] is False
    assert job["will_execute_write_bearing_action"] is False
    assert job["stdout_exists"] is True


def test_first_pass_resume_ui_smoke_job_is_allowlisted(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    payload = transcript_api.enqueue_app_smoke_job(
        state_root=state_root,
        job_type="first_pass_resume_ui_smoke",
        approval_token="RUN_APP_SMOKE_JOB",
        base_url="http://127.0.0.1:18876",
        start_thread=False,
    )

    job = payload["job"]
    job_record = json.loads(Path(job["path"]).read_text(encoding="utf-8"))

    assert payload["will_execute_arbitrary_shell"] is False
    assert payload["required_approval_token_checked"] == "RUN_APP_SMOKE_JOB"
    assert job["will_execute_external_action"] is True
    assert job["will_execute_write_bearing_action"] is False
    assert "smoke_first_pass_batch_resume_ui.py" in job_record["command"][1]
    assert "--cleanup" in job_record["command"]
    assert job_record["cleanup"] is True


def test_intelligence_smoke_job_tail_endpoint_is_path_confined(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    job_root = state_root / "smoke-jobs"
    job_root.mkdir(parents=True)
    stdout_path = job_root / "job.stdout.txt"
    stderr_path = job_root / "job.stderr.txt"
    stdout_path.write_text("line one\nline two\n", encoding="utf-8")
    stderr_path.write_text("failure detail\n", encoding="utf-8")
    (job_root / "job.json").write_text(
        json.dumps(
            {
                "job_id": "job",
                "job_type": "api_replay_smoke",
                "status": "failed",
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
            }
        ),
        encoding="utf-8",
    )
    (job_root / "escape.json").write_text(
        json.dumps(
            {
                "job_id": "escape",
                "job_type": "api_replay_smoke",
                "status": "failed",
                "stdout_path": str(tmp_path / "outside.txt"),
                "stderr_path": str(tmp_path / "outside.txt"),
            }
        ),
        encoding="utf-8",
    )
    server = transcript_api.TranscriptApiServer(
        ("127.0.0.1", 0),
        transcript_api.TranscriptApiHandler,
        store_root=tmp_path / "store",
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
        state_root=state_root,
        quiet=True,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        payload = json.loads(
            urlopen(
                f"http://{host}:{port}/api/intelligence/smoke-jobs/job/tail?stream=stdout&chars=8",
                timeout=5,
            ).read()
        )
        try:
            urlopen(f"http://{host}:{port}/api/intelligence/smoke-jobs/escape/tail?stream=stderr", timeout=5)
        except HTTPError as exc:
            assert exc.code == HTTPStatus.BAD_REQUEST
            assert "outside" in json.loads(exc.read())["error"]
        else:
            raise AssertionError("tail endpoint must reject paths outside smoke job directory")
    finally:
        server.shutdown()
        server.server_close()

    assert payload["schema_version"] == "transcribe-audio.app-smoke-job-tail.v1"
    assert payload["stream"] == "stdout"
    assert payload["tail"] == "ine two\n"
    assert payload["will_read_arbitrary_file"] is False
    assert payload["will_execute_write_bearing_action"] is False


def test_smoke_job_summary_exposes_known_evidence_paths(tmp_path: Path) -> None:
    browser_smoke_root = tmp_path / "state" / "browser-smokes"
    browser_smoke_root.mkdir(parents=True)
    report_path = browser_smoke_root / "resume.json"
    screenshot_path = browser_smoke_root / "resume.png"
    report_path.write_text("{}", encoding="utf-8")
    screenshot_path.write_bytes(b"png")
    job_path = tmp_path / "state" / "smoke-jobs" / "resume.json"
    transcript_api.write_app_smoke_job(
        job_path,
        {
            "job_id": "resume",
            "job_type": "first_pass_resume_ui_smoke",
            "status": "succeeded",
            "stdout_tail": "FIRST_PASS_RESUME_UI_SMOKE_JSON="
            + json.dumps(
                {
                    "schema_version": "transcribe-audio.first-pass-resume-ui-smoke.v1",
                    "status": "pass",
                    "report_path": str(report_path),
                    "screenshot_path": str(screenshot_path),
                    "checks": {"hasReviewQueue": True, "hasManifest": True},
                }
            )
            + "\n",
        },
    )

    summary = transcript_api.summarize_smoke_job(job_path)

    assert summary["evidence_summary"]["status"] == "pass"
    assert summary["evidence_summary"]["check_count"] == 2
    assert summary["evidence_summary"]["failed_check_count"] == 0
    assert "smoke-evidence?path=" in summary["evidence_summary"]["report_url"]
    assert str(report_path) not in summary["evidence_summary"]["report_url"]


def test_smoke_evidence_endpoint_is_path_confined(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    browser_smoke_root = state_root / "browser-smokes"
    browser_smoke_root.mkdir(parents=True)
    report_path = browser_smoke_root / "report.json"
    report_path.write_text('{"status":"pass"}', encoding="utf-8")
    outside_path = tmp_path / "outside.json"
    outside_path.write_text("{}", encoding="utf-8")
    bad_suffix = browser_smoke_root / "bad.txt"
    bad_suffix.write_text("nope", encoding="utf-8")
    server = transcript_api.TranscriptApiServer(
        ("127.0.0.1", 0),
        transcript_api.TranscriptApiHandler,
        store_root=tmp_path / "store",
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
        state_root=state_root,
        quiet=True,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        payload = urlopen(
            f"http://{host}:{port}/api/intelligence/smoke-evidence?path={quote(str(report_path))}",
            timeout=5,
        ).read()
        for path in (outside_path, bad_suffix):
            try:
                urlopen(
                    f"http://{host}:{port}/api/intelligence/smoke-evidence?path={quote(str(path))}",
                    timeout=5,
                )
            except HTTPError as exc:
                assert exc.code == HTTPStatus.BAD_REQUEST
            else:
                raise AssertionError("smoke evidence endpoint must reject unsafe paths")
    finally:
        server.shutdown()
        server.server_close()

    assert payload == b'{"status":"pass"}'


def test_cleanup_smoke_job_apply_requires_cleanup_token(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    try:
        transcript_api.enqueue_app_smoke_job(
            state_root=state_root,
            job_type="cleanup_smokes",
            approval_token="RUN_APP_SMOKE_JOB",
            base_url="http://127.0.0.1:18876",
            apply_cleanup=True,
            start_thread=False,
        )
    except ValueError as exc:
        assert "CLEANUP_APP_SMOKE_ARTIFACTS" in str(exc)
    else:
        raise AssertionError("cleanup apply must require the cleanup approval token")

    payload = transcript_api.enqueue_app_smoke_job(
        state_root=state_root,
        job_type="cleanup_smokes",
        approval_token="CLEANUP_APP_SMOKE_ARTIFACTS",
        base_url="http://127.0.0.1:18876",
        apply_cleanup=True,
        start_thread=False,
    )

    job = payload["job"]
    job_record = json.loads(Path(job["path"]).read_text(encoding="utf-8"))
    assert payload["required_approval_token_checked"] == "CLEANUP_APP_SMOKE_ARTIFACTS"
    assert job["will_execute_write_bearing_action"] is True
    assert job_record["apply_cleanup"] is True
    assert "--apply" in job_record["command"]


def test_smoke_job_write_is_atomic(tmp_path: Path) -> None:
    job_path = tmp_path / "state" / "smoke-jobs" / "job.json"
    transcript_api.write_app_smoke_job(job_path, {"job_id": "job", "status": "queued"})
    transcript_api.write_app_smoke_job(job_path, {"job_id": "job", "status": "running"})

    assert json.loads(job_path.read_text(encoding="utf-8"))["status"] == "running"
    assert not list(job_path.parent.glob("*.tmp"))


def test_cleanup_smoke_job_summary_is_exposed_from_stdout_tail(tmp_path: Path) -> None:
    job_path = tmp_path / "state" / "smoke-jobs" / "cleanup.json"
    cleanup_payload = {
        "schema_version": "transcribe-audio.app-smoke-cleanup.v1",
        "apply": False,
        "matched_run_count": 4,
        "kept_run_count": 1,
        "delete_run_count": 3,
        "matched_evidence_count": 12,
        "keep_evidence": 10,
        "evidence_days": 14,
        "delete_evidence_count": 2,
        "delete_run_paths": ["/not/exposed"],
        "delete_evidence_paths": ["/also/not/exposed"],
    }
    transcript_api.write_app_smoke_job(
        job_path,
        {
            "job_id": "cleanup",
            "job_type": "cleanup_smokes",
            "status": "succeeded",
            "stdout_tail": f"APP_SMOKE_CLEANUP_JSON={json.dumps(cleanup_payload)}\n",
        },
    )

    summary = transcript_api.summarize_smoke_job(job_path)

    assert summary["cleanup_summary"] == {
        "schema_version": "transcribe-audio.app-smoke-cleanup.v1",
        "apply": False,
        "matched_run_count": 4,
        "kept_run_count": 1,
        "delete_run_count": 3,
        "matched_evidence_count": 12,
        "keep_evidence": 10,
        "evidence_days": 14,
        "delete_evidence_count": 2,
    }


def test_cleanup_smoke_job_summary_tolerates_bad_count_fields(tmp_path: Path) -> None:
    job_path = tmp_path / "state" / "smoke-jobs" / "cleanup.json"
    cleanup_payload = {
        "schema_version": "transcribe-audio.app-smoke-cleanup.v1",
        "apply": False,
        "matched_run_count": "bad",
        "delete_evidence_count": None,
    }
    transcript_api.write_app_smoke_job(
        job_path,
        {
            "job_id": "cleanup",
            "job_type": "cleanup_smokes",
            "status": "succeeded",
            "stdout_tail": f"APP_SMOKE_CLEANUP_JSON={json.dumps(cleanup_payload)}\n",
        },
    )

    summary = transcript_api.summarize_smoke_job(job_path)

    assert summary["cleanup_summary"]["matched_run_count"] == 0
    assert summary["cleanup_summary"]["delete_evidence_count"] == 0


def test_intelligence_config_endpoint_returns_task_routing(tmp_path: Path) -> None:
    server = transcript_api.TranscriptApiServer(
        ("127.0.0.1", 0),
        transcript_api.TranscriptApiHandler,
        store_root=tmp_path / "store",
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
        state_root=tmp_path / "state",
        quiet=True,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        payload = json.loads(urlopen(f"http://{host}:{port}/api/intelligence/config", timeout=5).read())
    finally:
        server.shutdown()
        server.server_close()

    assert payload["schema_version"] == "transcribe-audio.intelligence-config.v1"
    assert "openai_readout" in payload["profiles"]
    assert payload["task_profiles"]["first_pass_summary"] == "openai_readout"
    assert payload["tasks"]["first_pass_summary"]["provider"] == "openai-compatible"
    assert payload["tasks"]["first_pass_summary"]["profile"] == "openai_readout"
    assert payload["tasks"]["app_supervisor"]["provider"] == "codex-app-server"


def test_intelligence_config_endpoint_exposes_auracall_agent_options(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("TRANSCRIPTS_INTELLIGENCE_CONFIG", raising=False)
    FakeAuraCallHandler.requests = []
    provider = ThreadingHTTPServer(("127.0.0.1", 0), FakeAuraCallHandler)
    provider_thread = threading.Thread(target=provider.serve_forever, daemon=True)
    provider_thread.start()
    host, provider_port = provider.server_address
    env_file = tmp_path / "auracall.env"
    env_file.write_text(
        "\n".join(
            [
                f"OPENAI_BASE_URL=http://{host}:{provider_port}/v1",
                "OPENAI_API_KEY=test-key",
                "AURACALL_AGENT_ID=transcripts-worker",
            ]
        ),
        encoding="utf-8",
    )
    server = transcript_api.TranscriptApiServer(
        ("127.0.0.1", 0),
        transcript_api.TranscriptApiHandler,
        store_root=tmp_path / "store",
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
        state_root=tmp_path / "state",
        batch_env_file=env_file,
        quiet=True,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        api_host, port = server.server_address
        payload = json.loads(urlopen(f"http://{api_host}:{port}/api/intelligence/config", timeout=5).read())
    finally:
        server.shutdown()
        server.server_close()
        provider.shutdown()
        provider.server_close()

    readiness = payload["auracall_readiness"]
    assert readiness["source"]["fetched"] is True
    assert readiness["selected_model"] == "agent:transcripts-worker"
    assert readiness["agent_options"][0]["id"] == "transcripts-worker"
    assert readiness["agent_options"][0]["model"] == "agent:transcripts-worker"
    assert readiness["agent_options"][0]["ready"] is True
    assert "project Transcripts" in readiness["agent_options"][0]["settings_description"]
    assert "test-key" not in json.dumps(readiness)


def test_intelligence_config_preview_and_apply_endpoints(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "intelligence.config.json"
    monkeypatch.setenv("TRANSCRIPTS_INTELLIGENCE_CONFIG", str(config_path))
    server = transcript_api.TranscriptApiServer(
        ("127.0.0.1", 0),
        transcript_api.TranscriptApiHandler,
        store_root=tmp_path / "store",
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
        state_root=tmp_path / "state",
        quiet=True,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        preview_request = Request(
            f"http://{host}:{port}/api/intelligence/config/preview",
            data=json.dumps(
                {
                    "task": "first_pass_summary",
                    "update": {"provider": "codex-exec", "model": "gpt-preview"},
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        preview = json.loads(urlopen(preview_request, timeout=5).read())
        assert preview["will_write"] is False
        assert not config_path.exists()

        blocked_request = Request(
            f"http://{host}:{port}/api/intelligence/config/apply",
            data=json.dumps(
                {
                    "task": "first_pass_summary",
                    "update": {"provider": "codex-exec"},
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urlopen(blocked_request, timeout=5)
        except HTTPError as exc:
            assert exc.code == 400
            assert "approval_token" in json.loads(exc.read())["error"]
        else:
            raise AssertionError("apply without approval token must fail")

        apply_request = Request(
            f"http://{host}:{port}/api/intelligence/config/apply",
            data=json.dumps(
                {
                    "task": "first_pass_summary",
                    "update": {"provider": "codex-exec", "model": "gpt-applied"},
                    "approval_token": "APPLY_INTELLIGENCE_CONFIG_UPDATE",
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        applied_response = urlopen(apply_request, timeout=5)
        applied = json.loads(applied_response.read())
        config = json.loads(urlopen(f"http://{host}:{port}/api/intelligence/config", timeout=5).read())
    finally:
        server.shutdown()
        server.server_close()

    assert applied_response.status == 202
    assert applied["applied"] is True
    assert config["tasks"]["first_pass_summary"]["provider"] == "codex-exec"
    assert config["tasks"]["first_pass_summary"]["model"] == "gpt-applied"


def test_automation_config_endpoint_defaults_preview_and_apply(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("TRANSCRIPTS_AUTOMATION_CONFIG", raising=False)
    state_root = tmp_path / "state"
    server = transcript_api.TranscriptApiServer(
        ("127.0.0.1", 0),
        transcript_api.TranscriptApiHandler,
        store_root=tmp_path / "store",
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
        state_root=state_root,
        quiet=True,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        defaults = json.loads(urlopen(f"http://{host}:{port}/api/automation/config", timeout=5).read())
        config_path = state_root / "automation.config.json"
        assert defaults["exists"] is False
        assert defaults["stages"]["initial_summary"]["enabled"] is False
        assert defaults["stages"]["initial_summary"]["mode"] == "manual"
        assert defaults["config_path"] == str(config_path)

        preview_request = Request(
            f"http://{host}:{port}/api/automation/config/preview",
            data=json.dumps(
                {
                    "update": {
                        "stages": {
                            "initial_summary": {
                                "enabled": True,
                                "mode": "one_click",
                                "requires_review": True,
                            }
                        }
                    }
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        preview = json.loads(urlopen(preview_request, timeout=5).read())
        assert preview["will_write"] is False
        assert preview["will_execute_workflow_stage"] is False
        assert preview["after"]["stages"]["initial_summary"]["enabled"] is True
        assert not config_path.exists()

        blocked_request = Request(
            f"http://{host}:{port}/api/automation/config/apply",
            data=json.dumps({"update": {"stages": {"initial_summary": {"enabled": True}}}}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urlopen(blocked_request, timeout=5)
        except HTTPError as exc:
            assert exc.code == 400
            assert "approval_token" in json.loads(exc.read())["error"]
        else:
            raise AssertionError("automation apply without approval token must fail")

        apply_request = Request(
            f"http://{host}:{port}/api/automation/config/apply",
            data=json.dumps(
                {
                    "update": {
                        "stages": {
                            "initial_summary": {
                                "enabled": True,
                                "mode": "one_click",
                                "requires_review": True,
                            }
                        }
                    },
                    "approval_token": "APPLY_AUTOMATION_CONFIG_UPDATE",
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        applied_response = urlopen(apply_request, timeout=5)
        applied = json.loads(applied_response.read())
        updated = json.loads(urlopen(f"http://{host}:{port}/api/automation/config", timeout=5).read())
    finally:
        server.shutdown()
        server.server_close()

    assert applied_response.status == 202
    assert applied["applied"] is True
    assert updated["exists"] is True
    assert updated["stages"]["initial_summary"]["enabled"] is True
    assert updated["stages"]["initial_summary"]["mode"] == "one_click"
    assert json.loads(config_path.read_text(encoding="utf-8"))["stages"]["initial_summary"]["enabled"] is True


def test_provenance_config_endpoint_redacts_and_applies_updates(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    state_root.mkdir()
    config_path = state_root / "provenance.config.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": "transcribe-audio.provenance-config.v1",
                "active_profile": "default",
                "profiles": {"default": {"source_ids": ["ical-private"]}},
                "sources": {
                    "ical-private": {
                        "kind": "ical_calendar",
                        "enabled": True,
                        "label": "Private calendar",
                        "url": "https://calendar.example.invalid/private-token",
                        "read_only": True,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    server = transcript_api.TranscriptApiServer(
        ("127.0.0.1", 0),
        transcript_api.TranscriptApiHandler,
        store_root=tmp_path / "store",
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
        state_root=state_root,
        quiet=True,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        payload = json.loads(urlopen(f"http://{host}:{port}/api/provenance/config", timeout=5).read())
        assert payload["exists"] is True
        assert "private-token" not in json.dumps(payload)

        preview_request = Request(
            f"http://{host}:{port}/api/provenance/config/preview",
            data=json.dumps(
                {
                    "update": {
                        "sources": {
                            "gws-work": {
                                "kind": "gws",
                                "enabled": True,
                                "label": "Work gws",
                                "read_only": True,
                            }
                        }
                    }
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        preview = json.loads(urlopen(preview_request, timeout=5).read())
        assert preview["will_write"] is False
        assert "gws-work" not in json.loads(config_path.read_text(encoding="utf-8"))["sources"]

        blocked_request = Request(
            f"http://{host}:{port}/api/provenance/config/apply",
            data=json.dumps({"update": {"active_profile": "default"}}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urlopen(blocked_request, timeout=5)
        except HTTPError as exc:
            assert exc.code == 400
            assert "approval_token" in json.loads(exc.read())["error"]
        else:
            raise AssertionError("apply without approval token must fail")

        apply_request = Request(
            f"http://{host}:{port}/api/provenance/config/apply",
            data=json.dumps(
                {
                    "approval_token": "APPLY_PROVENANCE_CONFIG_UPDATE",
                    "update": {
                        "sources": {
                            "gws-work": {
                                "kind": "gws",
                                "enabled": True,
                                "label": "Work gws",
                                "read_only": True,
                            }
                        }
                    },
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        applied_response = urlopen(apply_request, timeout=5)
        applied = json.loads(applied_response.read())
    finally:
        server.shutdown()
        server.server_close()

    assert applied_response.status == 202
    assert applied["applied"] is True
    assert "private-token" not in json.dumps(applied)
    assert "gws-work" in json.loads(config_path.read_text(encoding="utf-8"))["sources"]


def test_app_intelligence_run_prepare_and_read_endpoints(tmp_path: Path) -> None:
    server = transcript_api.TranscriptApiServer(
        ("127.0.0.1", 0),
        transcript_api.TranscriptApiHandler,
        store_root=tmp_path / "store",
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
        state_root=tmp_path / "state",
        quiet=True,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        request = Request(
            f"http://{host}:{port}/api/intelligence/runs/prepare",
            data=json.dumps(
                {
                    "workflow": "context-reread",
                    "purpose": "Prepare a supervised contextual reread.",
                    "document_id": "doc_123",
                    "run_id": "test-run",
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        prepared_response = urlopen(request, timeout=5)
        prepared = json.loads(prepared_response.read())
        listing = json.loads(urlopen(f"http://{host}:{port}/api/intelligence/runs", timeout=5).read())
        shown = json.loads(urlopen(f"http://{host}:{port}/api/intelligence/runs/test-run", timeout=5).read())
    finally:
        server.shutdown()
        server.server_close()

    assert prepared_response.status == 201
    assert prepared["run"]["run_id"] == "test-run"
    assert prepared["run"]["phase"] == "prepared"
    assert prepared["run"]["policy"]["host_owns_control_flow"] is True
    assert listing["total"] == 1
    assert shown["run"]["document_id"] == "doc_123"
    assert shown["events"][0]["event_type"] == "run_prepared"


def test_app_intelligence_session_start_preflight_endpoint_does_not_start_work(tmp_path: Path) -> None:
    server = transcript_api.TranscriptApiServer(
        ("127.0.0.1", 0),
        transcript_api.TranscriptApiHandler,
        store_root=tmp_path / "store",
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
        state_root=tmp_path / "state",
        quiet=True,
        codex_bin=sys.executable,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        prepare_request = Request(
            f"http://{host}:{port}/api/intelligence/runs/prepare",
            data=json.dumps(
                {
                    "workflow": "app-supervisor",
                    "purpose": "Prepare supervised work.",
                    "run_id": "preflight-run",
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urlopen(prepare_request, timeout=5).read()

        preflight_request = Request(
            f"http://{host}:{port}/api/intelligence/runs/preflight-run/session-start-preflight",
            data=json.dumps({"approval_token": "START_APP_SERVER_SESSION"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        preflight_response = urlopen(preflight_request, timeout=5)
        preflight = json.loads(preflight_response.read())

        append_request = Request(
            f"http://{host}:{port}/api/intelligence/runs/preflight-run/session-start-preflight",
            data=json.dumps(
                {
                    "approval_token": "APPEND_SESSION_START_PREFLIGHT_EVENT",
                    "append_event": True,
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        append_response = urlopen(append_request, timeout=5)
        appended = json.loads(append_response.read())
        shown = json.loads(urlopen(f"http://{host}:{port}/api/intelligence/runs/preflight-run", timeout=5).read())
    finally:
        server.shutdown()
        server.server_close()

    assert preflight_response.status == 200
    assert preflight["will_start_session"] is False
    assert preflight["checks"]["phase_prepared"] is True
    assert preflight["checks"]["approval_token_shape"] is True
    assert preflight["checks"]["provider_ready"] is False
    assert "provider_ready" in preflight["blocking_checks"]
    assert append_response.status == 202
    assert appended["event"]["event_type"] == "session_start_preflight"
    assert shown["events"][-1]["event_type"] == "session_start_preflight"


def test_app_intelligence_session_start_endpoint_starts_daemon_without_model_turn(tmp_path: Path, monkeypatch) -> None:
    commands: list[list[str]] = []

    monkeypatch.setattr(
        transcript_api,
        "codex_app_server_readiness",
        lambda codex_bin="codex": {"ready": True, "status": "ready"},
    )

    def fake_run_codex_command(args: list[str], *, timeout: int = 30) -> dict:
        commands.append(args)
        if args[-1] == "start":
            return {"args": args, "ok": True, "returncode": 0, "stdout": "started\n", "stderr": ""}
        return {"args": args, "ok": True, "returncode": 0, "stdout": '{"running": true}\n', "stderr": ""}

    monkeypatch.setattr(transcript_api, "run_codex_command", fake_run_codex_command)

    server = transcript_api.TranscriptApiServer(
        ("127.0.0.1", 0),
        transcript_api.TranscriptApiHandler,
        store_root=tmp_path / "store",
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
        state_root=tmp_path / "state",
        quiet=True,
        codex_bin="/usr/local/bin/codex",
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        prepare_request = Request(
            f"http://{host}:{port}/api/intelligence/runs/prepare",
            data=json.dumps(
                {
                    "workflow": "app-supervisor",
                    "purpose": "Prepare supervised work.",
                    "run_id": "session-run",
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urlopen(prepare_request, timeout=5).read()

        start_request = Request(
            f"http://{host}:{port}/api/intelligence/runs/session-run/session-start",
            data=json.dumps({"approval_token": "START_APP_SERVER_SESSION", "transport": "stdio"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        start_response = urlopen(start_request, timeout=5)
        started = json.loads(start_response.read())
        shown = json.loads(urlopen(f"http://{host}:{port}/api/intelligence/runs/session-run", timeout=5).read())
    finally:
        server.shutdown()
        server.server_close()

    assert start_response.status == 202
    assert started["ok"] is True
    assert started["will_start_model_turn"] is False
    assert commands == [
        ["/usr/local/bin/codex", "app-server", "daemon", "start"],
        ["/usr/local/bin/codex", "app-server", "daemon", "version"],
    ]
    assert shown["run"]["phase"] == "session_started"
    assert shown["run"]["state"]["active_codex_thread_id"] is None
    assert [event["event_type"] for event in shown["events"][-2:]] == [
        "app_server_session_start_requested",
        "app_server_session_started",
    ]


def test_app_intelligence_model_turn_preflight_endpoint_writes_prompt_packet(tmp_path: Path, monkeypatch) -> None:
    commands: list[list[str]] = []
    store_root = tmp_path / "store"
    ingest = transcript_store.ingest_artifact(
        write_transcript_artifact(tmp_path),
        root=store_root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
    )

    monkeypatch.setattr(
        transcript_api,
        "codex_app_server_readiness",
        lambda codex_bin="codex": {"ready": True, "status": "ready"},
    )

    def fake_run_codex_command(args: list[str], *, timeout: int = 30) -> dict:
        commands.append(args)
        return {"args": args, "ok": True, "returncode": 0, "stdout": "ok\n", "stderr": ""}

    monkeypatch.setattr(transcript_api, "run_codex_command", fake_run_codex_command)

    def fake_start_model_turn(**kwargs: object) -> dict:
        assert kwargs["codex_bin"] == "/usr/local/bin/codex"
        assert "Tempo Chemical samples" in str(kwargs["prompt_text"])
        assert kwargs["existing_thread_id"] == ""
        return {
            "ok": True,
            "thread_id": "thread_test",
            "turn_id": "turn_test",
            "thread_start_response": {"thread": {"id": "thread_test"}},
            "turn_start_response": {"turn": {"id": "turn_test"}},
            "events": [{"method": "turn/started", "params": {"turn": {"id": "turn_test"}}}],
        }

    monkeypatch.setattr(transcript_api.codex_app_server_client, "start_model_turn", fake_start_model_turn)

    def fake_inspect_model_turn(**kwargs: object) -> dict:
        assert kwargs["codex_bin"] == "/usr/local/bin/codex"
        assert kwargs["thread_id"] == "thread_test"
        assert kwargs["turn_id"] == "turn_test"
        return {
            "ok": True,
            "thread_id": "thread_test",
            "turn_id": "turn_test",
            "status": "completed",
            "completed": True,
            "output_text": json.dumps(
                {
                    "action": "ask_for_human_review",
                    "rationale": "Tempo readout requires operator review.",
                    "confidence": 0.8,
                    "review_flags": ["operator_review"],
                    "recommended_next_prompt": "Validate context sources.",
                }
            ),
            "thread_read_response": {"thread": {"id": "thread_test"}},
            "turns_list_response": {"data": [{"id": "turn_test", "status": "completed"}]},
            "items_list_response": {"data": []},
            "events": [{"method": "turn/completed", "params": {"turn": {"id": "turn_test"}}}],
        }

    monkeypatch.setattr(transcript_api.codex_app_server_client, "inspect_model_turn", fake_inspect_model_turn)

    server = transcript_api.TranscriptApiServer(
        ("127.0.0.1", 0),
        transcript_api.TranscriptApiHandler,
        store_root=store_root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
        state_root=tmp_path / "state",
        quiet=True,
        codex_bin="/usr/local/bin/codex",
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        prepare_request = Request(
            f"http://{host}:{port}/api/intelligence/runs/prepare",
            data=json.dumps(
                {
                    "workflow": "contextual_reread",
                    "purpose": "Prepare supervised work.",
                    "document_id": ingest.id,
                    "run_id": "packet-run",
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urlopen(prepare_request, timeout=5).read()
        start_request = Request(
            f"http://{host}:{port}/api/intelligence/runs/packet-run/session-start",
            data=json.dumps({"approval_token": "START_APP_SERVER_SESSION"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urlopen(start_request, timeout=5).read()

        packet_request = Request(
            f"http://{host}:{port}/api/intelligence/runs/packet-run/model-turn-preflight",
            data=json.dumps(
                {
                    "approval_token": "PREPARE_MODEL_TURN_PREFLIGHT",
                    "task": "contextual_reread",
                    "document_id": ingest.id,
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        packet_response = urlopen(packet_request, timeout=5)
        packet = json.loads(packet_response.read())
        review = json.loads(
            urlopen(
                f"http://{host}:{port}/api/intelligence/runs/packet-run/prompt-packets/{packet['packet']['packet_id']}",
                timeout=5,
            ).read()
        )
        send_preflight_request = Request(
            f"http://{host}:{port}/api/intelligence/runs/packet-run/prompt-packets/{packet['packet']['packet_id']}/send-preflight",
            data=json.dumps({"approval_token": "SEND_APP_SERVER_MODEL_TURN"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        send_preflight_response = urlopen(send_preflight_request, timeout=5)
        send_preflight = json.loads(send_preflight_response.read())
        send_request = Request(
            f"http://{host}:{port}/api/intelligence/runs/packet-run/prompt-packets/{packet['packet']['packet_id']}/send",
            data=json.dumps({"approval_token": "SEND_APP_SERVER_MODEL_TURN"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        send_response = urlopen(send_request, timeout=5)
        send = json.loads(send_response.read())
        status_request = Request(
            f"http://{host}:{port}/api/intelligence/runs/packet-run/turn-status",
            data=json.dumps({"approval_token": "CAPTURE_MODEL_TURN_STATUS"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        status_response = urlopen(status_request, timeout=5)
        status = json.loads(status_response.read())
        decision_request = Request(
            f"http://{host}:{port}/api/intelligence/runs/packet-run/structured-decision/validate",
            data=json.dumps({"approval_token": "VALIDATE_STRUCTURED_DECISION"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        decision_response = urlopen(decision_request, timeout=5)
        decision = json.loads(decision_response.read())
        apply_request = Request(
            f"http://{host}:{port}/api/intelligence/runs/packet-run/structured-decisions/{decision['decision_id']}/apply",
            data=json.dumps({"approval_token": "APPLY_STRUCTURED_DECISION", "reviewer": "api-test"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        apply_response = urlopen(apply_request, timeout=5)
        applied = json.loads(apply_response.read())
        human_review_request = Request(
            f"http://{host}:{port}/api/intelligence/runs/packet-run/structured-decisions/{decision['decision_id']}/human-review",
            data=json.dumps(
                {
                    "approval_token": "RECORD_HUMAN_REVIEW_DECISION",
                    "review_action": "resolve",
                    "reviewer": "api-test",
                    "note": "Reviewed in the API test.",
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        human_review_response = urlopen(human_review_request, timeout=5)
        human_review = json.loads(human_review_response.read())
        replay_manifest = json.loads(
            urlopen(f"http://{host}:{port}/api/intelligence/runs/packet-run/replay-manifest", timeout=5).read()
        )
        review_queue = json.loads(urlopen(f"http://{host}:{port}/api/review-queue?limit=20", timeout=5).read())
        shown = json.loads(urlopen(f"http://{host}:{port}/api/intelligence/runs/packet-run", timeout=5).read())
    finally:
        server.shutdown()
        server.server_close()

    assert packet_response.status == 201
    assert packet["will_send_prompt"] is False
    assert Path(packet["packet_path"]).exists()
    assert Path(packet["prompt_path"]).exists()
    assert packet["packet"]["document"]["id"] == ingest.id
    assert packet["packet"]["route"]["task"] == "contextual_reread"
    assert "Tempo Chemical samples" in Path(packet["prompt_path"]).read_text(encoding="utf-8")
    assert review["will_send_prompt"] is False
    assert review["future_required_approval_token_for_send"] == "SEND_APP_SERVER_MODEL_TURN"
    assert "Tempo Chemical samples" in review["prompt_text"]
    assert send_preflight_response.status == 200
    assert send_preflight["ok"] is True
    assert send_preflight["will_send_prompt"] is False
    assert send_preflight["will_write_event"] is False
    assert send_preflight["prompt_char_count"] > 0
    assert send_response.status == 202
    assert send["ok"] is True
    assert send["codex_thread_id"] == "thread_test"
    assert send["codex_turn_id"] == "turn_test"
    assert send["will_execute_downstream_action"] is False
    assert shown["run"]["prompt_packets"][0]["sent"] is True
    assert status_response.status == 202
    assert status["completed"] is True
    assert status["will_execute_structured_decision"] is False
    assert status["codex_thread_id"] == "thread_test"
    assert status["codex_turn_id"] == "turn_test"
    assert Path(status["artifact_path"]).exists()
    assert decision_response.status == 202
    assert decision["valid"] is True
    assert decision["decision"]["action"] == "ask_for_human_review"
    assert decision["will_execute_host_action"] is False
    assert apply_response.status == 202
    assert applied["decision_action"] == "ask_for_human_review"
    assert applied["applied_ledger_state"] is True
    assert applied["will_execute_external_action"] is False
    assert applied["will_execute_write_bearing_action"] is False
    assert human_review_response.status == 202
    assert human_review["review_action"] == "resolve"
    assert human_review["human_review_status"] == "resolved"
    app_review_bucket = next(bucket for bucket in review_queue["buckets"] if bucket["id"] == "app_intelligence_human_review")
    assert app_review_bucket["count"] == 0
    assert any(item["status"] == "resolved" for item in review_queue["items"])
    assert shown["run"]["phase"] == "human_review_requested"
    assert shown["run"]["status"] == "needs_human_review"
    assert shown["run"]["state"]["active_codex_thread_id"] == "thread_test"
    assert shown["run"]["decisions"][0]["status"] == "applied"
    assert shown["run"]["decisions"][0]["human_review"]["status"] == "resolved"
    assert shown["events"][-1]["event_type"] == "human_review_decision_recorded"
    assert shown["codex_events_count"] == 2
    assert replay_manifest["schema_version"] == "transcribe-audio.app-intelligence-replay-manifest.v1"
    assert replay_manifest["will_execute_write_bearing_action"] is False
    assert replay_manifest["will_read_artifact_content"] is False
    assert [item["artifact_role"] for item in replay_manifest["artifacts"]] == [
        "prompt_packet_json",
        "prompt_text",
        "model_turn_status",
        "structured_decision_validation",
        "structured_decision_apply",
    ]
    assert all(item["can_read_via_artifact_endpoint"] for item in replay_manifest["artifacts"])


def test_app_intelligence_fork_preflight_endpoint_is_preview_only(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    app_intelligence_ledger.create_run(
        state_root=state_root,
        workflow="contextual_reread",
        purpose="Preview branch alternatives.",
        run_id="fork-api-run",
    )
    run_payload = app_intelligence_ledger.response_for_run(state_root=state_root, run_id="fork-api-run")["run"]
    decision_dir = state_root / "app-intelligence-runs" / "fork-api-run" / "artifacts" / "structured-decisions"
    decision_dir.mkdir(parents=True)
    decision_path = decision_dir / "decision-fork.json"
    decision_path.write_text(
        json.dumps(
            {
                "schema_version": "transcribe-audio.app-intelligence-structured-decision-validation.v1",
                "decision_id": "decision-fork",
                "run_id": "fork-api-run",
                "valid": True,
                "decision": {
                    "action": "fork_branches",
                    "rationale": "Explore two context strategies.",
                    "confidence": 0.82,
                    "review_flags": [],
                    "branch_count": 2,
                    "experiments": ["Drive-heavy context", "Graph-heavy context"],
                },
            }
        ),
        encoding="utf-8",
    )
    run_payload["phase"] = "model_turn_completed"
    run_payload["decisions"] = [
        {
            "decision_id": "decision-fork",
            "valid": True,
            "action": "fork_branches",
            "status": "validated",
            "artifact_path": str(decision_path),
            "will_execute_host_action": False,
            "created_at": "2026-05-20T12:00:00Z",
        }
    ]
    (state_root / "app-intelligence-runs" / "fork-api-run" / "run.json").write_text(
        json.dumps(run_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    server = transcript_api.TranscriptApiServer(
        ("127.0.0.1", 0),
        transcript_api.TranscriptApiHandler,
        store_root=tmp_path / "store",
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
        state_root=state_root,
        quiet=True,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        preflight_request = Request(
            f"http://{host}:{port}/api/intelligence/runs/fork-api-run/structured-decisions/decision-fork/fork-preflight",
            data=json.dumps({"approval_token": "PREVIEW_FORK_BRANCHES", "reviewer": "api-test"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        preflight_response = urlopen(preflight_request, timeout=5)
        preflight = json.loads(preflight_response.read())
        shown = json.loads(urlopen(f"http://{host}:{port}/api/intelligence/runs/fork-api-run", timeout=5).read())
    finally:
        server.shutdown()
        server.server_close()

    assert preflight_response.status == 202
    assert preflight["planned_branches"][0]["branch_id"].startswith("main-fork-1")
    assert preflight["will_create_thread"] is False
    assert preflight["will_modify_branches"] is False
    assert preflight["will_run_provider"] is False
    assert Path(preflight["artifact_path"]).exists()
    assert shown["run"]["phase"] == "model_turn_completed"
    assert shown["run"]["decisions"][0]["status"] == "validated"
    assert shown["events"][-1]["event_type"] == "fork_branches_preflight"


def test_app_intelligence_continue_apply_endpoint_is_ledger_only(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    app_intelligence_ledger.create_run(
        state_root=state_root,
        workflow="contextual_reread",
        purpose="Continue current branch.",
        run_id="continue-api-run",
    )
    run_payload = app_intelligence_ledger.response_for_run(state_root=state_root, run_id="continue-api-run")["run"]
    decision_dir = state_root / "app-intelligence-runs" / "continue-api-run" / "artifacts" / "structured-decisions"
    decision_dir.mkdir(parents=True)
    decision_path = decision_dir / "decision-continue.json"
    decision_path.write_text(
        json.dumps(
            {
                "schema_version": "transcribe-audio.app-intelligence-structured-decision-validation.v1",
                "decision_id": "decision-continue",
                "run_id": "continue-api-run",
                "valid": True,
                "decision": {
                    "action": "continue_current_branch",
                    "rationale": "The current branch has enough context for the next turn.",
                    "confidence": 0.88,
                    "review_flags": [],
                    "recommended_next_prompt": "Continue with the current route.",
                },
            }
        ),
        encoding="utf-8",
    )
    run_payload["phase"] = "model_turn_completed"
    run_payload["status"] = "running"
    run_payload["decisions"] = [
        {
            "decision_id": "decision-continue",
            "valid": True,
            "action": "continue_current_branch",
            "status": "validated",
            "artifact_path": str(decision_path),
            "will_execute_host_action": False,
            "created_at": "2026-05-20T12:00:00Z",
        }
    ]
    (state_root / "app-intelligence-runs" / "continue-api-run" / "run.json").write_text(
        json.dumps(run_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    server = transcript_api.TranscriptApiServer(
        ("127.0.0.1", 0),
        transcript_api.TranscriptApiHandler,
        store_root=tmp_path / "store",
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
        state_root=state_root,
        quiet=True,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        apply_request = Request(
            f"http://{host}:{port}/api/intelligence/runs/continue-api-run/structured-decisions/decision-continue/apply",
            data=json.dumps({"approval_token": "APPLY_STRUCTURED_DECISION", "reviewer": "api-test"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        apply_response = urlopen(apply_request, timeout=5)
        applied = json.loads(apply_response.read())
        artifact_response = urlopen(
            f"http://{host}:{port}/api/intelligence/runs/continue-api-run/artifacts?path={quote(applied['artifact_path'], safe='')}",
            timeout=5,
        )
        artifact = json.loads(artifact_response.read())
        try:
            urlopen(
                f"http://{host}:{port}/api/intelligence/runs/continue-api-run/artifacts?path={quote(str(tmp_path / 'unregistered.json'), safe='')}",
                timeout=5,
            )
        except HTTPError as exc:
            assert exc.code == 400
        else:
            raise AssertionError("Expected unregistered artifact paths to be rejected.")
        shown = json.loads(urlopen(f"http://{host}:{port}/api/intelligence/runs/continue-api-run", timeout=5).read())
    finally:
        server.shutdown()
        server.server_close()

    assert apply_response.status == 202
    assert artifact_response.status == 200
    assert artifact["artifact_type"] == "json"
    assert artifact["json"]["decision_id"] == "decision-continue"
    assert artifact["will_execute_write_bearing_action"] is False
    assert applied["decision_action"] == "continue_current_branch"
    assert applied["will_execute_write_bearing_action"] is False
    assert applied["will_fork_or_rollback"] is False
    assert Path(applied["artifact_path"]).exists()
    assert shown["run"]["phase"] == "current_branch_continued"
    assert shown["run"]["status"] == "running"
    assert shown["run"]["final"] is None
    assert shown["run"]["latest_continuation"]["current_branch"] == "main"
    assert shown["run"]["decisions"][0]["status"] == "applied"
    assert shown["events"][-1]["event_type"] == "structured_decision_applied"


def test_app_intelligence_rollback_preflight_endpoint_is_preview_only(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    app_intelligence_ledger.create_run(
        state_root=state_root,
        workflow="contextual_reread",
        purpose="Preview rollback.",
        run_id="rollback-api-run",
    )
    run_payload = app_intelligence_ledger.response_for_run(state_root=state_root, run_id="rollback-api-run")["run"]
    decision_dir = state_root / "app-intelligence-runs" / "rollback-api-run" / "artifacts" / "structured-decisions"
    decision_dir.mkdir(parents=True)
    decision_path = decision_dir / "decision-rollback.json"
    decision_path.write_text(
        json.dumps(
            {
                "schema_version": "transcribe-audio.app-intelligence-structured-decision-validation.v1",
                "decision_id": "decision-rollback",
                "run_id": "rollback-api-run",
                "valid": True,
                "decision": {
                    "action": "rollback",
                    "rationale": "Return to the last stable reviewed state.",
                    "confidence": 0.74,
                    "review_flags": ["rollback_preview"],
                    "target_branch": "main",
                    "target_event_id": "event_before_bad_context",
                    "target_turn_id": "turn_before_bad_context",
                },
            }
        ),
        encoding="utf-8",
    )
    run_payload["phase"] = "model_turn_completed"
    run_payload["decisions"] = [
        {
            "decision_id": "decision-rollback",
            "valid": True,
            "action": "rollback",
            "status": "validated",
            "artifact_path": str(decision_path),
            "will_execute_host_action": False,
            "created_at": "2026-05-20T12:00:00Z",
        }
    ]
    (state_root / "app-intelligence-runs" / "rollback-api-run" / "run.json").write_text(
        json.dumps(run_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    server = transcript_api.TranscriptApiServer(
        ("127.0.0.1", 0),
        transcript_api.TranscriptApiHandler,
        store_root=tmp_path / "store",
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
        state_root=state_root,
        quiet=True,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        preflight_request = Request(
            f"http://{host}:{port}/api/intelligence/runs/rollback-api-run/structured-decisions/decision-rollback/rollback-preflight",
            data=json.dumps({"approval_token": "PREVIEW_ROLLBACK", "reviewer": "api-test"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        preflight_response = urlopen(preflight_request, timeout=5)
        preflight = json.loads(preflight_response.read())
        artifact_response = urlopen(
            f"http://{host}:{port}/api/intelligence/runs/rollback-api-run/artifacts?path={quote(preflight['artifact_path'], safe='')}",
            timeout=5,
        )
        artifact = json.loads(artifact_response.read())
        shown = json.loads(urlopen(f"http://{host}:{port}/api/intelligence/runs/rollback-api-run", timeout=5).read())
    finally:
        server.shutdown()
        server.server_close()

    assert preflight_response.status == 202
    assert artifact_response.status == 200
    assert artifact["json"]["decision_id"] == "decision-rollback"
    assert artifact["will_execute_write_bearing_action"] is False
    assert preflight["target_branch"] == "main"
    assert preflight["target_event_id"] == "event_before_bad_context"
    assert preflight["target_turn_id"] == "turn_before_bad_context"
    assert preflight["will_modify_branches"] is False
    assert preflight["will_revert_artifacts"] is False
    assert preflight["will_run_provider"] is False
    assert Path(preflight["artifact_path"]).exists()
    assert shown["run"]["phase"] == "model_turn_completed"
    assert shown["run"]["decisions"][0]["status"] == "validated"
    assert shown["events"][-1]["event_type"] == "rollback_preflight"


def test_batch_status_counts_prefers_provider_aggregate_counts() -> None:
    assert transcript_api.batch_status_counts(
        {
            "counts": {"total": 5, "in_progress": 5, "completed": 0},
            "jobs": [{"status": "in_progress"} for _ in range(5)],
        }
    ) == {"total": 5, "in_progress": 5, "completed": 0}


def test_prepare_first_pass_summary_endpoint_writes_dry_run_manifest(tmp_path: Path) -> None:
    store_root = tmp_path / "store"
    state_root = tmp_path / "state"
    env_file = tmp_path / "auracall.env"
    env_file.write_text(
        "\n".join(
            [
                "OPENAI_BASE_URL=http://127.0.0.1:18095/v1",
                "OPENAI_API_KEY=test-key",
                "AURACALL_BATCH_URL=http://127.0.0.1:18095/v1/response-batches",
                "AURACALL_DISPATCH_TEAM=transcribe-audio-chatgpt-pro-pool",
                "AURACALL_DISPATCH_MODEL=gpt-5.2-pro",
            ]
        ),
        encoding="utf-8",
    )
    transcript_store.ingest_artifact(
        write_transcript_artifact(tmp_path),
        root=store_root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
    )
    server = transcript_api.TranscriptApiServer(
        ("127.0.0.1", 0),
        transcript_api.TranscriptApiHandler,
        store_root=store_root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
        state_root=state_root,
        batch_env_file=env_file,
        quiet=True,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        request = Request(
            f"http://{host}:{port}/api/review-queue/first-pass-summaries/prepare",
            data=json.dumps({"limit": 1}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        response = urlopen(request, timeout=5)
        payload = json.loads(response.read())
        manifest = json.loads(Path(payload["manifest"]).read_text(encoding="utf-8"))

        assert response.status == 201
        assert payload["action"] == "prepare_first_pass_summary_batch"
        assert payload["bucket"] == "first_pass_summaries"
        assert payload["request_count"] == 1
        assert payload["dry_run"] is True
        assert payload["batch_id"] is None
        assert payload["workflow"] == "transcribe-audio-first-pass-summary"
        assert payload["artifact_file"] == "first_pass_readout.json"
        assert manifest["batch"] is None
        assert manifest["request_count"] == 1
        assert manifest["dispatch_team"] == "transcribe-audio-chatgpt-pro-pool"
        assert manifest["model"] == "gpt-5.2-pro"
        assert manifest["batch_payload"]["metadata"]["workflow"] == "transcribe-audio-first-pass-summary"
        assert manifest["batch_payload"]["dispatch"]["team"] == "transcribe-audio-chatgpt-pro-pool"
    finally:
        server.shutdown()
        server.server_close()


def test_first_pass_summary_submit_and_status_use_prepared_manifest(tmp_path: Path) -> None:
    store_root = tmp_path / "store"
    state_root = tmp_path / "state"
    env_file = tmp_path / "auracall.env"
    transcript_store.ingest_artifact(
        write_transcript_artifact(tmp_path),
        root=store_root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
    )
    FakeAuraCallHandler.requests = []
    provider = ThreadingHTTPServer(("127.0.0.1", 0), FakeAuraCallHandler)
    provider_thread = threading.Thread(target=provider.serve_forever, daemon=True)
    provider_thread.start()
    host, provider_port = provider.server_address
    env_file.write_text(
        "\n".join(
            [
                f"OPENAI_BASE_URL=http://{host}:{provider_port}/v1",
                "OPENAI_API_KEY=test-key",
                f"AURACALL_BATCH_URL=http://{host}:{provider_port}/v1/response-batches",
                "AURACALL_DISPATCH_TEAM=transcribe-audio-chatgpt-pro-pool",
                "AURACALL_DISPATCH_MODEL=gpt-5.2-pro",
            ]
        ),
        encoding="utf-8",
    )
    server = transcript_api.TranscriptApiServer(
        ("127.0.0.1", 0),
        transcript_api.TranscriptApiHandler,
        store_root=store_root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
        state_root=state_root,
        batch_env_file=env_file,
        quiet=True,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        prepare_request = Request(
            f"http://{host}:{port}/api/review-queue/first-pass-summaries/prepare",
            data=json.dumps({"limit": 1}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        prepared = json.loads(urlopen(prepare_request, timeout=5).read())
        manifest_path = Path(prepared["manifest"])

        blocked_request = Request(
            f"http://{host}:{port}/api/review-queue/first-pass-summaries/submit",
            data=json.dumps({"manifest": prepared["manifest"]}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urlopen(blocked_request, timeout=5)
        except HTTPError as exc:
            assert exc.code == 400
            assert "approval_token" in json.loads(exc.read())["error"]
        else:
            raise AssertionError("Submit without approval token must fail")

        submit_request = Request(
            f"http://{host}:{port}/api/review-queue/first-pass-summaries/submit",
            data=json.dumps(
                {
                    "manifest": prepared["manifest"],
                    "approval_token": "SUBMIT_FIRST_PASS_SUMMARY_BATCH",
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        submitted_response = urlopen(submit_request, timeout=5)
        submitted = json.loads(submitted_response.read())
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        assert submitted_response.status == 202
        assert submitted["status"] == "submitted"
        assert submitted["batch_id"] == "batch_test"
        assert submitted["dry_run"] is False
        assert manifest["batch"]["id"] == "batch_test"
        assert manifest["dry_run"] is False
        assert FakeAuraCallHandler.requests[0]["dispatch"]["team"] == "transcribe-audio-chatgpt-pro-pool"
        assert FakeAuraCallHandler.requests[0]["metadata"]["workflow"] == "transcribe-audio-first-pass-summary"

        status_request = Request(
            f"http://{host}:{port}/api/review-queue/first-pass-summaries/status",
            data=json.dumps({"manifest": prepared["manifest"]}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        status_payload = json.loads(urlopen(status_request, timeout=5).read())

        assert status_payload["status"] == "running"
        assert status_payload["batch_id"] == "batch_test"
        assert status_payload["batch_counts"] == {"running": 1}
        assert json.loads(manifest_path.read_text(encoding="utf-8"))["last_status"]["status"] == "running"
    finally:
        server.shutdown()
        server.server_close()
        provider.shutdown()
        provider.server_close()


def test_first_pass_summary_manifests_endpoint_lists_redacted_summaries(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    manifest_root = state_root / "first-pass-summary-batches"
    manifest_root.mkdir(parents=True)
    manifest_path = manifest_root / "first-pass-summary-prepare-test.json"
    transcript_api.write_json_file(
        manifest_path,
        {
            "request_count": 2,
            "dry_run": False,
            "batch": {"id": "batch_visible"},
            "last_status": {"status": "running", "counts": {"running": 2}},
            "materialized": [{"document_id": "doc1"}],
            "materialization_errors": [{"request_id": "req2"}],
            "batch_payload": {
                "metadata": {"workflow": "transcribe-audio-first-pass-summary"},
                "requests": [{"input": [{"content": "private transcript text"}]}],
            },
        },
    )
    server = transcript_api.TranscriptApiServer(
        ("127.0.0.1", 0),
        transcript_api.TranscriptApiHandler,
        store_root=tmp_path / "store",
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
        state_root=state_root,
        quiet=True,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        payload = json.loads(
            urlopen(
                f"http://{host}:{port}/api/review-queue/first-pass-summaries/manifests?limit=5",
                timeout=5,
            ).read()
        )
    finally:
        server.shutdown()
        server.server_close()

    assert payload["schema_version"] == "transcribe-audio.first-pass-summary-batch-manifests.v1"
    assert payload["will_read_request_payloads"] is False
    assert payload["will_read_transcript_content"] is False
    assert payload["total"] == 1
    item = payload["items"][0]
    assert item["manifest"] == str(manifest_path)
    assert item["batch_id"] == "batch_visible"
    assert item["status"] == "running"
    assert item["batch_counts"] == {"running": 2}
    assert item["materialized_count"] == 1
    assert item["materialization_error_count"] == 1
    assert "private transcript text" not in json.dumps(item)
