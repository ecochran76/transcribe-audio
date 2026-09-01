from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from urllib.request import Request, urlopen

import pytest

import transcript_store
import transcript_api
from conversation_evidence_fabric import EvidenceAnchor, EvidenceFabric, EvidenceRequest
from conversation_knowledge_evidence import EvidenceScope
from conversation_knowledge_store import ConversationKnowledgeStore
from identity_review_workflow import IdentityReviewWorkflow
from mail_hypothesis_review import (
    MailHypothesisProjectionError,
    StaleMailHypothesisReview,
    install_mail_hypothesis_source,
    load_mail_hypothesis_projection,
)


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _write_artifacts(root: Path) -> tuple[Path, dict[str, object]]:
    artifact_root = root / "pilot"
    hypotheses_dir = artifact_root / "hypotheses"
    hypotheses_dir.mkdir(parents=True)
    hypothesis = {
        "schema_version": "transcribe-audio.mail-relationship-hypothesis.v1",
        "hypothesis_id": "mail-hypothesis-1",
        "hypothesis_kind": "correspondence",
        "relationship_type": "CORRESPONDS_WITH",
        "subject_contact_id": "contact:contact-alex",
        "counterpart_type": "contact_candidate",
        "counterpart_id": "contact:contact-morgan",
        "counterpart_label": "Morgan Example",
        "directionality": "symmetric",
        "status": "proposed",
        "basis": "Independent mail metadata observations meet the frozen threshold.",
        "why_not_accepted": "Mail exchange does not prove relationship meaning.",
        "observation_count": 3,
        "independent_thread_count": 3,
        "first_observed_at": "2025-01-01T12:00:00Z",
        "last_observed_at": "2025-03-01T12:00:00Z",
        "evidence_observation_ids": ["mail-observation-1", "mail-observation-2"],
        "evidence_independence_group_ids": ["mail-thread-1", "mail-thread-2"],
        "conflicts": [],
        "effect_counts": {"accepted_relationships": 0},
    }
    artifact = {
        "schema_version": "transcribe-audio.plan0073-p5-shadow-hypotheses.v1",
        "conversation_id": "conversation-origin",
        "input_watermark": "fixture-watermark",
        "hypotheses": [hypothesis],
        "excluded_reason_counts": {},
        "effects": {"accepted_relationships": 0},
    }
    name = "conversation-origin.json"
    (hypotheses_dir / name).write_text(_canonical(artifact), encoding="utf-8")
    aggregate = {
        "schema_version": "transcribe-audio.plan0073-p5-execution-receipt.v1",
        "preview_id": "plan0073-p5-fixture",
        "status": "complete",
        "counts": {"hypotheses": 1},
        "effects": {
            "accepted_relationships": 0,
            "accepted_roles": 0,
            "biometric_effects": 0,
            "graphiti_writes": 0,
            "person_merges": 0,
            "provider_writes": 0,
            "speaker_assignments": 0,
        },
        "artifacts": {
            "hypotheses": [{"name": name, "content_sha256": _sha(artifact)}]
        },
    }
    aggregate["content_sha256"] = _sha(aggregate)
    (artifact_root / "aggregate-validation.json").write_text(
        _canonical(aggregate), encoding="utf-8"
    )
    return artifact_root, hypothesis


def _workflow(tmp_path: Path) -> tuple[IdentityReviewWorkflow, Path, dict[str, object]]:
    ConversationKnowledgeStore(tmp_path).migrate(backup=False)
    for contact_id, label in (
        ("contact-alex", "Alex Example"),
        ("contact-morgan", "Morgan Example"),
    ):
        with transcript_store.connect(tmp_path) as con:
            transcript_store.init_db(con)
            con.execute(
                """
                INSERT INTO contacts (
                  id, label, email, external_ref, metadata_json, created_at, updated_at
                ) VALUES (?, ?, '', '', ?, '2025-03-01T12:00:00Z', '2025-03-01T12:00:00Z')
                """,
                (
                    contact_id,
                    label,
                    json.dumps(
                        {
                            "contact_class": "person_candidate",
                            "calendar_attendee": {"appearances": []},
                        }
                    ),
                ),
            )
            con.commit()
    artifact_root, hypothesis = _write_artifacts(tmp_path)
    install_mail_hypothesis_source(tmp_path, artifact_root)
    return IdentityReviewWorkflow(tmp_path), artifact_root, hypothesis


def _submission(
    *,
    action: str,
    expected_version: str,
    source_hash: str,
    key: str,
    decided_at: str,
) -> dict[str, str]:
    return {
        "schema_version": "transcribe-audio.mail-hypothesis-review-submission.v1",
        "hypothesis_id": "mail-hypothesis-1",
        "action": action,
        "expected_projection_version": expected_version,
        "source_content_sha256": source_hash,
        "reviewer": "operator:fixture",
        "decided_at": decided_at,
        "idempotency_key": key,
        "note": "Reviewed against the available metadata.",
    }


def test_hash_pinned_projection_attaches_mail_hypothesis_to_both_contacts(
    tmp_path: Path,
) -> None:
    workflow, _, _ = _workflow(tmp_path)

    payload = workflow.list_people(kind="local_contact", limit=10)

    assert payload["graph_discovery"]["mail_hypothesis_count"] == 1
    assert payload["graph_discovery"]["mail_source"]["status"] == "ready"
    by_name = {item["primary_name"]: item for item in payload["items"]}
    alex = by_name["Alex Example"]["relationship_hypotheses"][0]
    morgan = by_name["Morgan Example"]["relationship_hypotheses"][0]
    assert alex["hypothesis_id"] == morgan["hypothesis_id"] == "mail-hypothesis-1"
    assert alex["mail_direction"] == morgan["mail_direction"] == "symmetric"
    assert alex["review_state"] == "unreviewed"
    assert alex["projection_version"] == "1"
    assert alex["evidence_source"] == "mail_metadata"


def test_projection_rejects_hash_drift(tmp_path: Path) -> None:
    _, artifact_root, _ = _workflow(tmp_path)
    artifact_path = artifact_root / "hypotheses" / "conversation-origin.json"
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    payload["hypotheses"][0]["observation_count"] = 99
    artifact_path.write_text(_canonical(payload), encoding="utf-8")

    with pytest.raises(MailHypothesisProjectionError, match="hash"):
        load_mail_hypothesis_projection(tmp_path)


def test_invalid_locator_fails_closed_without_emptying_contacts(tmp_path: Path) -> None:
    workflow, _, _ = _workflow(tmp_path)
    locator_path = transcript_store.store_dir(tmp_path) / "mail-hypothesis-source.json"
    locator = json.loads(locator_path.read_text(encoding="utf-8"))
    locator["hypothesis_count"] = "not-an-integer"
    locator_path.write_text(_canonical(locator), encoding="utf-8")

    payload = workflow.list_people(kind="local_contact", limit=10)

    assert payload["total"] == 2
    assert payload["graph_discovery"]["mail_hypothesis_count"] == 0
    assert payload["graph_discovery"]["mail_source"] == {
        "status": "invalid",
        "reason_code": "configured_mail_hypothesis_source_failed_validation",
    }


def test_defer_is_durable_idempotent_and_stale_safe(tmp_path: Path) -> None:
    workflow, _, _ = _workflow(tmp_path)
    source_hash = load_mail_hypothesis_projection(tmp_path).source_content_sha256
    submission = _submission(
        action="defer",
        expected_version="1",
        source_hash=source_hash,
        key="mail-review-defer-1",
        decided_at="2026-09-01T12:00:00Z",
    )

    first = workflow.record_mail_hypothesis_review(submission)
    replay = workflow.record_mail_hypothesis_review(submission)

    assert first["idempotent_replay"] is False
    assert replay["idempotent_replay"] is True
    assert first["projection_version"] == "2"
    assert first["accepted_relationship_effect_count"] == 0
    item = workflow.list_people(kind="local_contact", limit=10)["items"][0]
    assert item["relationship_hypotheses"][0]["review_state"] == "deferred"
    changed_replay = dict(submission, note="Different decision content.")
    with pytest.raises(MailHypothesisProjectionError, match="idempotency"):
        workflow.record_mail_hypothesis_review(changed_replay)
    with pytest.raises(StaleMailHypothesisReview):
        workflow.record_mail_hypothesis_review(
            _submission(
                action="reject",
                expected_version="1",
                source_hash=source_hash,
                key="mail-review-stale-1",
                decided_at="2026-09-01T12:01:00Z",
            )
        )


def test_accept_enters_ledger_and_shared_fabric_only_after_review(tmp_path: Path) -> None:
    workflow, _, _ = _workflow(tmp_path)
    projection = load_mail_hypothesis_projection(tmp_path)
    fabric = EvidenceFabric(tmp_path)
    request = EvidenceRequest(
        purpose="conversation_understanding",
        conversation_id="conversation-later",
        anchors=(EvidenceAnchor("person", "contact:contact-alex"),),
        query_terms=(),
        scopes=(EvidenceScope("local-knowledge", "local", "local"),),
        capabilities=("accepted_relationships",),
        as_of="2026-09-02T12:00:00Z",
        hindsight_policy="exclude",
        allowed_freshness_states=("current",),
        max_records=10,
        max_characters=20_000,
        max_provider_calls=0,
        max_relationship_hops=1,
    )

    assert fabric.collect(request).relationships == ()
    receipt = workflow.record_mail_hypothesis_review(
        _submission(
            action="accept",
            expected_version="1",
            source_hash=projection.source_content_sha256,
            key="mail-review-accept-1",
            decided_at="2026-09-01T12:00:00Z",
        )
    )

    assert receipt["accepted_relationship_effect_count"] == 1
    accepted = fabric.collect(request).relationships
    assert len(accepted) == 1
    assert accepted[0].relationship_type == "CORRESPONDS_WITH"
    assert accepted[0].subject_id == "contact:contact-alex"
    assert accepted[0].object_id == "contact:contact-morgan"
    row = workflow.list_people(kind="local_contact", limit=10)["items"][0][
        "relationship_hypotheses"
    ][0]
    assert row["review_state"] == "accepted"
    assert row["projection_version"] == "2"


def test_reject_never_enters_accepted_relationship_projection(tmp_path: Path) -> None:
    workflow, _, _ = _workflow(tmp_path)
    source_hash = load_mail_hypothesis_projection(tmp_path).source_content_sha256
    workflow.record_mail_hypothesis_review(
        _submission(
            action="reject",
            expected_version="1",
            source_hash=source_hash,
            key="mail-review-reject-1",
            decided_at="2026-09-01T12:00:00Z",
        )
    )

    with transcript_store.connect(tmp_path) as con:
        count = con.execute(
            "SELECT COUNT(*) FROM knowledge_identity_relationship_projection"
        ).fetchone()[0]
    assert count == 0


def test_reject_after_accept_removes_relationship_from_accepted_fabric(
    tmp_path: Path,
) -> None:
    workflow, _, _ = _workflow(tmp_path)
    source_hash = load_mail_hypothesis_projection(tmp_path).source_content_sha256
    workflow.record_mail_hypothesis_review(
        _submission(
            action="accept",
            expected_version="1",
            source_hash=source_hash,
            key="mail-review-accept-before-reject",
            decided_at="2026-09-01T12:00:00Z",
        )
    )
    receipt = workflow.record_mail_hypothesis_review(
        _submission(
            action="reject",
            expected_version="2",
            source_hash=source_hash,
            key="mail-review-reject-after-accept",
            decided_at="2026-09-01T12:01:00Z",
        )
    )

    assert receipt["projection_version"] == "3"
    with transcript_store.connect(tmp_path) as con:
        status = con.execute(
            "SELECT status FROM knowledge_identity_relationship_projection"
        ).fetchone()[0]
    assert status == "rejected"
    row = workflow.list_people(kind="local_contact", limit=10)["items"][0][
        "relationship_hypotheses"
    ][0]
    assert row["review_state"] == "rejected"


def test_http_review_endpoint_records_a_bounded_decision(tmp_path: Path) -> None:
    workflow, _, _ = _workflow(tmp_path)
    source_hash = load_mail_hypothesis_projection(tmp_path).source_content_sha256
    server = transcript_api.TranscriptApiServer(
        ("127.0.0.1", 0),
        transcript_api.TranscriptApiHandler,
        store_root=workflow.root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
        state_root=tmp_path / "state",
        quiet=True,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        submission = _submission(
            action="defer",
            expected_version="1",
            source_hash=source_hash,
            key="mail-review-http-1",
            decided_at="2026-09-01T12:00:00Z",
        )
        request = Request(
            f"http://{host}:{port}/api/relationship-hypotheses/mail-hypothesis-1/reviews",
            data=json.dumps(submission).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        response = urlopen(request, timeout=5)
        receipt = json.loads(response.read())
    finally:
        server.shutdown()
        server.server_close()

    assert response.status == 201
    assert receipt["action"] == "defer"
    assert receipt["accepted_relationship_effect_count"] == 0
