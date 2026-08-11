from __future__ import annotations

import json
from pathlib import Path

import context_sources
import conversation_identity_policy
import conversation_identity_retrieval
import conversation_knowledge_evidence
import conversation_knowledge_store
import pytest
import transcript_artifact_access


def test_policy_uses_only_explicit_retrieval_source_identity_and_scope() -> None:
    resolved = {
        "gws": [
            context_sources.GwsProvenanceConfig(
                enabled=True,
                profile_label="Display label must not become scope",
                config_dir=Path("/explicit/gws"),
                include_calendar_details=False,
                include_drive_search=False,
                include_gmail_search=True,
                include_people_contacts=True,
                include_other_contacts=True,
            )
        ],
        "odollo": [
            context_sources.OdolloProvenanceConfig(
                enabled=True,
                profiles=("soylei-prod",),
                command=("odollo",),
                repo_root=Path("/explicit/odollo"),
                config_path=Path("/explicit/odollo.yml"),
                include_contacts=True,
                include_leads=True,
                include_log_notes=True,
            )
        ],
        "source_contexts": [
            {"source_id": "gws-default", "relationship_scope": "personal"},
            {"source_id": "odollo-soylei", "relationship_scope": "company"},
        ],
        "retrieval_sources": [
            {
                "source_id": "gws-default",
                "source_profile_id": "gws-default",
                "provider_kind": "gws",
                "account_id": "owner@example.com",
                "tenant_id": "",
                "evidence_capabilities": ["people", "gmail"],
            },
            {
                "source_id": "odollo-soylei",
                "source_profile_id": "odollo-soylei",
                "provider_kind": "odollo",
                "account_id": "",
                "tenant_id": "soylei-prod",
                "evidence_capabilities": ["contacts", "leads", "log_notes"],
            },
        ],
        "warnings": ["bounded warning"],
    }

    built = conversation_identity_policy.build_identity_evidence_policy(
        resolved,
        requested_at="2026-07-29T18:00:00Z",
        request_id="request-1",
        run_id="run-1",
        environment={"PATH": "/bin"},
        prepared_query_terms=("person@example.com", "Project Orchard"),
    )

    assert [
        (
            scope.source_profile_id,
            scope.account_id,
            scope.tenant_id,
        )
        for scope in built.policy.scopes
    ] == [
        ("gws-default", "owner@example.com", ""),
        ("odollo-soylei", "", "soylei-prod"),
    ]
    assert built.policy.capabilities == (
        "people",
        "gmail",
        "contacts",
        "leads",
        "log_notes",
    )
    assert [
        (
            adapter.scope
            if hasattr(adapter, "scope")
            else adapter.config.scope
        ).source_profile_id
        for adapter in built.policy.provider_adapters
    ] == [
        "gws-default",
        "odollo-soylei",
    ]
    assert built.policy.requested_at == "2026-07-29T18:00:00Z"
    assert built.policy.request_id == "request-1"
    assert built.policy.run_id == "run-1"
    assert built.policy.hindsight_policy == "allow_later_retrieved"
    assert built.policy.prepared_query_terms == (
        "person@example.com",
        "Project Orchard",
    )
    assert built.source_contexts == tuple(resolved["source_contexts"])
    assert built.warnings == ("bounded warning",)


def test_discovery_citations_map_to_durable_projected_utterance_ids() -> None:
    labels, clue_ids = conversation_identity_policy.discovery_retrieval_inputs(
        {
            "speaker_clues": [
                {
                    "speaker_label": "Speaker A",
                    "transcript_clue_ids": ["utterance-2", "utterance-1"],
                }
            ],
            "speaker_group_hints": [
                {
                    "speaker_labels": ["Speaker A", "Speaker B"],
                    "transcript_clue_ids": ["utterance-2"],
                }
            ],
        },
        utterance_ids=("durable-1", "durable-2", "durable-3"),
        default_speaker_labels=("Speaker A", "Speaker B"),
    )

    assert labels == ("Speaker A", "Speaker B")
    assert clue_ids == ("durable-2", "durable-1")


def test_discovery_provider_terms_include_person_hints_and_model_terms() -> None:
    terms = conversation_identity_policy.discovery_provider_terms(
        {
            "speaker_clues": [
                {
                    "person_hints": [
                        {
                            "email": "person@example.com",
                            "name": "Example Person",
                            "organization": "Example Co",
                        }
                    ],
                    "retrieval_terms": [
                        "Project Orchard",
                        "person@example.com",
                    ],
                }
            ],
            "conversation_clues": [
                {"retrieval_terms": ["orchard catalyst"]}
            ],
        }
    )

    assert terms == (
        "person@example.com",
        "Example Person",
        "Example Co",
        "Project Orchard",
        "orchard catalyst",
    )


def test_legacy_transcript_materializes_deterministic_private_identity_snapshot(
    tmp_path: Path,
) -> None:
    transcript_path = tmp_path / "legacy.transcript.json"
    transcript_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "transcript_title": "Legacy review",
                "source_media_path": "/recordings/original-name.m4a",
                "transcript_text": "Hello.",
                "utterances": [
                    {"speaker": "A", "start": 0, "end": 1000, "text": "Hello."}
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    source_bytes = transcript_path.read_bytes()

    first = transcript_artifact_access.materialize_private_transcript_identity_snapshot(
        transcript_path,
        document_id="document-legacy-1",
        state_root=tmp_path / "state",
    )
    second = transcript_artifact_access.materialize_private_transcript_identity_snapshot(
        transcript_path,
        document_id="document-legacy-1",
        state_root=tmp_path / "state",
    )

    assert transcript_path.read_bytes() == source_bytes
    assert first == second
    assert first.source_was_derived is True
    assert first.path != transcript_path
    assert first.path.stat().st_mode & 0o777 == 0o600
    assert first.path.parent.stat().st_mode & 0o777 == 0o700
    snapshot = json.loads(first.path.read_text(encoding="utf-8"))
    assert snapshot["schema_version"] == 2
    assert snapshot["conversation_id"]
    assert snapshot["recording_id"]


def test_transcript_preparation_mirrors_reviewed_people_into_shadow_scope(
    tmp_path: Path,
) -> None:
    transcript_path = tmp_path / "legacy.transcript.json"
    transcript_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "transcript_title": "Reviewed roster test",
                "transcript_text": "Hello.",
                "utterances": [
                    {"speaker": "A", "start": 0, "end": 1000, "text": "Hello."}
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    source_bytes = transcript_path.read_bytes()
    person_id = "00000000-0000-4000-8000-000000000921"
    source_store = conversation_knowledge_store.ConversationKnowledgeStore(
        tmp_path / "source-store"
    )
    source_store.migrate(backup=False)
    source_store.save_person_snapshot(
        conversation_knowledge_store.PersonSnapshot(
            person=conversation_knowledge_store.PersonRecord(
                person_id=person_id,
                status="reviewed",
                primary_name="Reviewed Person",
            ),
            source_records=(
                conversation_knowledge_store.SourceRecord(
                    source_record_id="review-source-1",
                    person_id=person_id,
                    source_profile_id="plan0062-human-review",
                    provider_kind="human_review",
                    account_id="",
                    tenant_id="",
                    external_ref="reviewed-person-1",
                    label="Reviewed Person",
                    relationship_scope="speaker_identity",
                    identifier_authority="operator_review",
                    observed_at="2026-08-09T12:00:00Z",
                    content_hash="a" * 64,
                ),
            ),
        )
    )

    prepared = conversation_identity_policy.prepare_transcript_identity_evidence(
        transcript_path,
        {"speaker_clues": [], "conversation_clues": []},
        state_root=tmp_path / "state",
        source_store_root=tmp_path / "source-store",
        document_id="document-reviewed-roster",
        resolved={
            "gws": [],
            "odollo": [],
            "source_contexts": [],
            "retrieval_sources": [],
            "warnings": [],
        },
        environment={"PATH": "/bin"},
        requested_at="2026-08-11T12:00:00Z",
    )

    assert transcript_path.read_bytes() == source_bytes
    assert prepared.preparation_transcript_path != transcript_path
    assert prepared.bundle.request.prepared_person_ids == (person_id,)
    assert prepared.bundle.people[0].display_name == "Reviewed Person"
    assert prepared.bundle.people[0].source_profile_ids == (
        "plan0062-human-review",
    )
    assert [scope.source_profile_id for scope in prepared.bundle.request.scopes] == [
        "plan0062-human-review"
    ]


def test_transcript_preparation_projects_privately_and_freezes_request_before_adapter(
    tmp_path: Path,
    monkeypatch,
) -> None:
    transcript_path = tmp_path / "meeting.transcript.json"
    transcript_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "conversation_id": "00000000-0000-4000-8000-000000000901",
                "recording_id": "00000000-0000-4000-8000-000000000902",
                "transcript_title": "Orchard review",
                "recording_start": "2026-07-29T10:00:00-05:00",
                "recording_end": "2026-07-29T10:05:00-05:00",
                "transcript_text": "Hello orchard.",
                "utterances": [
                    {
                        "speaker": "A",
                        "start": 0,
                        "end": 1000,
                        "text": "Hello orchard.",
                    }
                ],
                "event": {
                    "summary": "Orchard review",
                    "attendees": [
                        {
                            "displayName": "Example Person",
                            "email": "person@example.com",
                        }
                    ],
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    source_bytes = transcript_path.read_bytes()
    adapter_requests = []

    class EmptyAdapter:
        adapter_id = "empty-adapter-v1"

        def retrieve(self, request):
            adapter_requests.append(request)
            return conversation_identity_retrieval.ProviderRetrievalResult()

    policy = conversation_identity_retrieval.IdentityEvidencePolicy(
        scopes=(
            conversation_knowledge_evidence.EvidenceScope(
                source_profile_id="gws-default",
                account_id="owner@example.com",
                tenant_id="",
            ),
        ),
        capabilities=("people",),
        provider_adapters=(EmptyAdapter(),),
        request_id="00000000-0000-4000-8000-000000000903",
        requested_at="2026-07-29T18:00:00Z",
    )
    policy_build = conversation_identity_policy.IdentityEvidencePolicyBuild(
        policy=policy,
        source_contexts=({"source_id": "gws-default"},),
        retrieval_sources=({"source_id": "gws-default"},),
        warnings=(),
    )
    monkeypatch.setattr(
        conversation_identity_policy,
        "build_identity_evidence_policy",
        lambda *args, **kwargs: policy_build,
    )

    prepared = conversation_identity_policy.prepare_transcript_identity_evidence(
        transcript_path,
        {
            "speaker_clues": [
                {
                    "speaker_label": "A",
                    "transcript_clue_ids": ["utterance-1"],
                }
            ]
        },
        state_root=tmp_path / "state",
        resolved={
            "gws": [],
            "odollo": [],
            "source_contexts": [],
            "retrieval_sources": [],
            "warnings": [],
        },
        requested_at="2026-07-29T18:00:00Z",
    )

    assert transcript_path.read_bytes() == source_bytes
    assert not (tmp_path / "meeting.processing.json").exists()
    assert len(adapter_requests) == 1
    assert adapter_requests[0].query_terms[:2] == (
        "person@example.com",
        "Example Person",
    )
    assert prepared.bundle.request.clue_ids != ("utterance-1",)
    assert prepared.bundle.persisted_bundle.status == "complete"
    assert Path(prepared.projection_receipt.receipt_path).is_file()
    assert prepared.retrieval_receipt_path.is_file()
    assert prepared.retrieval_receipt_path.stat().st_mode & 0o777 == 0o600
    retrieval_receipt = json.loads(
        prepared.retrieval_receipt_path.read_text(encoding="utf-8")
    )
    assert retrieval_receipt["request_sha256"]
    assert retrieval_receipt["query_plan_sha256"]
    assert retrieval_receipt["bundle_sha256"] == (
        prepared.bundle.persisted_bundle.content_hash
    )
    assert prepared.shadow_root.stat().st_mode & 0o777 == 0o700


def test_legacy_rollback_requires_explicit_token_and_writes_private_receipt(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="approval token"):
        conversation_identity_policy.record_legacy_rollback(
            state_root=tmp_path,
            document_id="document-1",
            operator="operator",
            approval_token="",
        )

    receipt = conversation_identity_policy.record_legacy_rollback(
        state_root=tmp_path,
        document_id="document-1",
        operator="operator",
        approval_token=(
            conversation_identity_policy.LEGACY_ROLLBACK_APPROVAL_TOKEN
        ),
    )

    receipt_path = Path(receipt["receipt_path"])
    assert receipt_path.is_file()
    assert receipt_path.stat().st_mode & 0o777 == 0o600
    assert receipt_path.parent.stat().st_mode & 0o777 == 0o700
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert payload["warning"] == (
        conversation_identity_policy.LEGACY_ROLLBACK_WARNING
    )
    assert "approval_token" not in payload
