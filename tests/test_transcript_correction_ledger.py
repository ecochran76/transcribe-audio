from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import conversation_knowledge_store
from transcript_correction_ledger import (
    TerminologyEntrySpec,
    TranscriptCorrectionLedger,
)


def _ledger(tmp_path: Path) -> TranscriptCorrectionLedger:
    store = conversation_knowledge_store.ConversationKnowledgeStore(tmp_path)
    store.migrate(backup=False)
    return TranscriptCorrectionLedger(tmp_path)


def _a2_fixture(name: str) -> dict[str, object]:
    path = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "dev"
        / "fixtures"
        / "plan-0072-a2"
        / name
    )
    return json.loads(path.read_text(encoding="utf-8"))


def test_terminology_scope_precedence_and_equal_scope_conflicts(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)
    entries = (
        TerminologyEntrySpec(
            entry_id="term-global-ciso",
            canonical_term="CISO",
            expansion="Chief Information Security Officer",
            definition="A security leadership role.",
            aliases=("C.I.S.O.",),
            asr_confusions=(),
            pronunciation_hints=("see-so",),
            scope_type="global",
            scope_id="global",
            status="reviewed",
        ),
        TerminologyEntrySpec(
            entry_id="term-domain-seso",
            canonical_term="SESO",
            expansion="semi-epoxidized soybean oil",
            definition="A chemistry material term.",
            aliases=(),
            asr_confusions=("CISO",),
            pronunciation_hints=("see-so",),
            scope_type="domain",
            scope_id="chemistry",
            status="reviewed",
        ),
        TerminologyEntrySpec(
            entry_id="term-project-seso",
            canonical_term="SoyLei SESO",
            expansion="semi-epoxidized soybean oil",
            definition="The project-specific display form.",
            aliases=(),
            asr_confusions=("CISO",),
            pronunciation_hints=(),
            scope_type="project_matter",
            scope_id="project-soylei",
            status="reviewed",
        ),
        TerminologyEntrySpec(
            entry_id="term-conversation-a",
            canonical_term="SESO batch A",
            expansion="",
            definition="First proposed conversation-local form.",
            aliases=(),
            asr_confusions=("CISO",),
            pronunciation_hints=(),
            scope_type="conversation",
            scope_id="conversation-1",
            status="reviewed",
        ),
        TerminologyEntrySpec(
            entry_id="term-conversation-b",
            canonical_term="SESO batch B",
            expansion="",
            definition="Conflicting conversation-local form.",
            aliases=(),
            asr_confusions=("CISO",),
            pronunciation_hints=(),
            scope_type="conversation",
            scope_id="conversation-1",
            status="reviewed",
        ),
    )
    receipt = ledger.register_terminology(
        version="terms-v1",
        entries=entries,
        status="reviewed",
        created_at="2026-08-16T14:00:00Z",
    )

    global_resolution = ledger.resolve_terminology(
        "CISO",
        terminology_version_id=receipt.terminology_version_id,
        context={},
    )
    domain_resolution = ledger.resolve_terminology(
        "CISO",
        terminology_version_id=receipt.terminology_version_id,
        context={"domain": "chemistry"},
    )
    project_resolution = ledger.resolve_terminology(
        "CISO",
        terminology_version_id=receipt.terminology_version_id,
        context={"domain": "chemistry", "project_matter": "project-soylei"},
    )
    conflict = ledger.resolve_terminology(
        "CISO",
        terminology_version_id=receipt.terminology_version_id,
        context={
            "conversation": "conversation-1",
            "project_matter": "project-soylei",
            "domain": "chemistry",
        },
    )

    assert global_resolution.status == "resolved"
    assert global_resolution.canonical_term == "CISO"
    assert global_resolution.scope_type == "global"
    assert domain_resolution.canonical_term == "SESO"
    assert domain_resolution.match_kind == "asr_confusion"
    assert project_resolution.canonical_term == "SoyLei SESO"
    assert project_resolution.scope_type == "project_matter"
    assert conflict.status == "review_required"
    assert conflict.canonical_term == ""
    assert conflict.candidate_entry_ids == (
        "term-conversation-a",
        "term-conversation-b",
    )
    assert conflict.reason_code == "equal_scope_conflict"


def test_terminology_hints_are_reviewed_scoped_and_version_pinned(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)
    reviewed = ledger.register_terminology(
        version="terms-hints-v1",
        entries=(
            TerminologyEntrySpec(
                entry_id="term-hint-global",
                canonical_term="CISO",
                expansion="Chief Information Security Officer",
                definition="A global security role.",
                aliases=("C.I.S.O.",),
                asr_confusions=(),
                pronunciation_hints=("see-so",),
                scope_type="global",
                scope_id="global",
                status="reviewed",
            ),
            TerminologyEntrySpec(
                entry_id="term-hint-domain",
                canonical_term="SESO",
                expansion="semi-epoxidized soybean oil",
                definition="A chemistry material term.",
                aliases=(),
                asr_confusions=("CISO",),
                pronunciation_hints=("see-so",),
                scope_type="domain",
                scope_id="chemistry",
                status="reviewed",
            ),
            TerminologyEntrySpec(
                entry_id="term-hint-draft-entry",
                canonical_term="Unreviewed",
                expansion="",
                definition="Not eligible for provider hints.",
                aliases=(),
                asr_confusions=(),
                pronunciation_hints=(),
                scope_type="global",
                scope_id="global",
                status="draft",
            ),
        ),
        status="reviewed",
        created_at="2026-08-16T14:05:00Z",
    )

    hints = ledger.terminology_hints(
        terminology_version_id=reviewed.terminology_version_id,
        context={"domain": "chemistry"},
    )

    assert hints.terminology_version_id == reviewed.terminology_version_id
    assert hints.version == "terms-hints-v1"
    assert hints.content_hash == reviewed.content_hash
    assert tuple(hint.entry_id for hint in hints.hints) == (
        "term-hint-domain",
        "term-hint-global",
    )
    assert hints.hints[0].asr_confusions == ("CISO",)
    assert all(hint.entry_id != "term-hint-draft-entry" for hint in hints.hints)


def test_accepted_span_correction_creates_non_destructive_normalized_generation(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)
    raw_text = "We discussed CISO quality. CISO remains unresolved."
    raw = ledger.record_raw_transcript(
        conversation_id="00000000-0000-4000-8000-000000000201",
        recording_id="00000000-0000-4000-8000-000000000202",
        source_artifact_sha256="a" * 64,
        transcript_text=raw_text,
        utterances=(
            {
                "speaker": "SPEAKER_1",
                "start_ms": 0,
                "end_ms": 3200,
                "text": raw_text,
            },
        ),
        captured_at="2026-08-15T18:00:00Z",
        created_at="2026-08-16T14:10:00Z",
    )
    first_start = raw_text.index("CISO")
    first = ledger.propose_correction(
        raw_generation_id=raw.raw_generation_id,
        span_start=first_start,
        span_end=first_start + len("CISO"),
        replacement_text="SESO",
        correction_kind="asr_confusion",
        terminology_entry_id="",
        scope={"type": "domain", "id": "chemistry"},
        evidence_ids=("evidence-1",),
        confidence=0.98,
        correction_pass="pre_identity",
        processing_version="correction-v1",
        cascade_count=0,
        created_at="2026-08-16T14:11:00Z",
    )
    second_start = raw_text.rindex("CISO")
    deferred = ledger.propose_correction(
        raw_generation_id=raw.raw_generation_id,
        span_start=second_start,
        span_end=second_start + len("CISO"),
        replacement_text="SESO",
        correction_kind="asr_confusion",
        terminology_entry_id="",
        scope={"type": "domain", "id": "chemistry"},
        evidence_ids=("evidence-2",),
        confidence=0.60,
        correction_pass="pre_identity",
        processing_version="correction-v1",
        cascade_count=0,
        created_at="2026-08-16T14:12:00Z",
    )
    ledger.decide_correction(
        proposal_id=first.proposal_id,
        action="accept",
        reviewer="reviewer:test",
        method="fixture_review",
        decided_at="2026-08-16T14:13:00Z",
        idempotency_key="accept-first-ciso",
    )
    ledger.decide_correction(
        proposal_id=deferred.proposal_id,
        action="defer",
        reviewer="reviewer:test",
        method="fixture_review",
        decided_at="2026-08-16T14:14:00Z",
        idempotency_key="defer-second-ciso",
    )

    normalized = ledger.normalize(
        raw_generation_id=raw.raw_generation_id,
        processing_version="correction-v1",
        correction_pass="pre_identity",
        context={"domain": "chemistry"},
        terminology_version_id="",
        index_version="transcript-layers-v1",
        created_at="2026-08-16T14:15:00Z",
    )

    assert normalized.status == "inserted"
    assert normalized.accepted_correction_ids == (first.proposal_id,)
    assert normalized.normalized_text == (
        "We discussed SESO quality. CISO remains unresolved."
    )
    assert normalized.normalized_transcript_sha256 == hashlib.sha256(
        normalized.normalized_text.encode("utf-8")
    ).hexdigest()
    raw_after = ledger.load_raw_generation(raw.raw_generation_id)
    assert raw_after["transcript_text"] == raw_text
    assert raw_after["transcript_sha256"] == hashlib.sha256(
        raw_text.encode("utf-8")
    ).hexdigest()
    loaded = ledger.load_normalized_generation(normalized.normalized_generation_id)
    assert loaded["raw_generation_id"] == raw.raw_generation_id
    assert loaded["normalized_text"] == normalized.normalized_text
    assert loaded["accepted_correction_ids"] == [first.proposal_id]
    assert loaded["raw_to_normalized_map"][0] == {
        "raw_start": 0,
        "raw_end": first_start,
        "normalized_start": 0,
        "normalized_end": first_start,
        "kind": "unchanged",
    }


def test_search_indexes_raw_and_selected_normalized_layers_with_provenance(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)
    raw_text = "The CISO material needs testing."
    raw = ledger.record_raw_transcript(
        conversation_id="00000000-0000-4000-8000-000000000211",
        recording_id="00000000-0000-4000-8000-000000000212",
        source_artifact_sha256="b" * 64,
        transcript_text=raw_text,
        utterances=(
            {
                "speaker": "SPEAKER_1",
                "start_ms": 0,
                "end_ms": 1800,
                "text": raw_text,
            },
        ),
        captured_at="2026-08-15T19:00:00Z",
        created_at="2026-08-16T14:20:00Z",
    )
    start = raw_text.index("CISO")
    proposal = ledger.propose_correction(
        raw_generation_id=raw.raw_generation_id,
        span_start=start,
        span_end=start + 4,
        replacement_text="SESO",
        correction_kind="asr_confusion",
        terminology_entry_id="",
        scope={"type": "domain", "id": "chemistry"},
        evidence_ids=("evidence-search",),
        confidence=0.99,
        correction_pass="pre_identity",
        processing_version="correction-v1",
        cascade_count=0,
        created_at="2026-08-16T14:21:00Z",
    )
    ledger.decide_correction(
        proposal_id=proposal.proposal_id,
        action="accept",
        reviewer="reviewer:test",
        method="fixture_review",
        decided_at="2026-08-16T14:22:00Z",
        idempotency_key="accept-search-correction",
    )
    normalized = ledger.normalize(
        raw_generation_id=raw.raw_generation_id,
        processing_version="correction-v1",
        correction_pass="pre_identity",
        context={"domain": "chemistry"},
        terminology_version_id="",
        index_version="transcript-layers-v1",
        created_at="2026-08-16T14:23:00Z",
    )

    shared = ledger.search_transcripts("material testing")
    raw_only = ledger.search_transcripts("CISO")
    normalized_only = ledger.search_transcripts("SESO")

    assert {(item.layer, item.generation_id) for item in shared} == {
        ("raw", raw.raw_generation_id),
        ("normalized", normalized.normalized_generation_id),
    }
    assert [(item.layer, item.generation_id) for item in raw_only] == [
        ("raw", raw.raw_generation_id)
    ]
    assert [(item.layer, item.generation_id) for item in normalized_only] == [
        ("normalized", normalized.normalized_generation_id)
    ]
    assert all(item.conversation_id.endswith("211") for item in shared)
    assert all(item.recording_id.endswith("212") for item in shared)
    reindex = ledger.load_reindex_receipt(normalized.normalized_generation_id)
    assert reindex["raw_generation_id"] == raw.raw_generation_id
    assert reindex["normalized_generation_id"] == normalized.normalized_generation_id
    assert reindex["index_version"] == "transcript-layers-v1"
    assert reindex["indexed_layer_count"] == 2
    assert reindex["raw_transcript_sha256"] != reindex[
        "normalized_transcript_sha256"
    ]


def test_semantic_map_requires_exact_normalized_and_raw_span_lineage(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)
    raw_text = "We discussed CISO quality."
    raw = ledger.record_raw_transcript(
        conversation_id="00000000-0000-4000-8000-000000000221",
        recording_id="00000000-0000-4000-8000-000000000222",
        source_artifact_sha256="c" * 64,
        transcript_text=raw_text,
        utterances=(
            {
                "speaker": "SPEAKER_1",
                "start_ms": 0,
                "end_ms": 1400,
                "text": raw_text,
            },
        ),
        captured_at="2026-08-15T20:00:00Z",
        created_at="2026-08-16T14:30:00Z",
    )
    start = raw_text.index("CISO")
    proposal = ledger.propose_correction(
        raw_generation_id=raw.raw_generation_id,
        span_start=start,
        span_end=start + 4,
        replacement_text="SESO",
        correction_kind="asr_confusion",
        terminology_entry_id="",
        scope={"type": "domain", "id": "chemistry"},
        evidence_ids=("evidence-semantic",),
        confidence=0.99,
        correction_pass="pre_identity",
        processing_version="correction-v1",
        cascade_count=0,
        created_at="2026-08-16T14:31:00Z",
    )
    ledger.decide_correction(
        proposal_id=proposal.proposal_id,
        action="accept",
        reviewer="reviewer:test",
        method="fixture_review",
        decided_at="2026-08-16T14:32:00Z",
        idempotency_key="accept-semantic-correction",
    )
    normalized = ledger.normalize(
        raw_generation_id=raw.raw_generation_id,
        processing_version="correction-v1",
        correction_pass="pre_identity",
        context={"domain": "chemistry"},
        terminology_version_id="",
        index_version="transcript-layers-v1",
        created_at="2026-08-16T14:33:00Z",
    )
    normalized_span_text = "SESO quality"
    normalized_start = normalized.normalized_text.index(normalized_span_text)
    raw_span_text = "CISO quality"
    raw_start = raw_text.index(raw_span_text)
    sections = {
        "topics": [
            {
                "label": "material quality",
                "normalized_span": {
                    "start": normalized_start,
                    "end": normalized_start + len(normalized_span_text),
                    "text_sha256": hashlib.sha256(
                        normalized_span_text.encode("utf-8")
                    ).hexdigest(),
                },
                "raw_lineage": [
                    {
                        "raw_generation_id": raw.raw_generation_id,
                        "start": raw_start,
                        "end": raw_start + len(raw_span_text),
                        "text_sha256": hashlib.sha256(
                            raw_span_text.encode("utf-8")
                        ).hexdigest(),
                    }
                ],
            }
        ],
        "terms": [],
        "entities": [],
        "questions": [],
    }

    receipt = ledger.record_semantic_map(
        normalized_generation_id=normalized.normalized_generation_id,
        sections=sections,
        created_at="2026-08-16T14:34:00Z",
    )
    loaded = ledger.load_semantic_map(receipt.semantic_map_id)

    assert receipt.status == "inserted"
    assert loaded["transcript_only"] is True
    assert loaded["sections"] == sections
    assert loaded["normalized_generation_id"] == normalized.normalized_generation_id

    invalid = {
        **sections,
        "topics": [{**sections["topics"][0], "raw_lineage": []}],
    }
    with pytest.raises(ValueError, match="raw lineage"):
        ledger.record_semantic_map(
            normalized_generation_id=normalized.normalized_generation_id,
            sections=invalid,
            created_at="2026-08-16T14:35:00Z",
        )


def test_two_pass_and_one_identity_cascade_controller_stops_for_manual_review(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)
    raw_text = "Morgan described the project timeline."
    raw = ledger.record_raw_transcript(
        conversation_id="00000000-0000-4000-8000-000000000231",
        recording_id="00000000-0000-4000-8000-000000000232",
        source_artifact_sha256="d" * 64,
        transcript_text=raw_text,
        utterances=(
            {
                "speaker": "SPEAKER_1",
                "start_ms": 0,
                "end_ms": 1700,
                "text": raw_text,
            },
        ),
        captured_at="2026-08-15T21:00:00Z",
        created_at="2026-08-16T14:40:00Z",
    )

    pre = ledger.normalize(
        raw_generation_id=raw.raw_generation_id,
        processing_version="correction-v1",
        correction_pass="pre_identity",
        context={},
        terminology_version_id="",
        index_version="transcript-layers-v1",
        created_at="2026-08-16T14:41:00Z",
    )
    post = ledger.normalize(
        raw_generation_id=raw.raw_generation_id,
        processing_version="correction-v1",
        correction_pass="post_identity",
        context={},
        terminology_version_id="",
        index_version="transcript-layers-v1",
        created_at="2026-08-16T14:42:00Z",
    )
    replay = ledger.normalize(
        raw_generation_id=raw.raw_generation_id,
        processing_version="correction-v1",
        correction_pass="post_identity",
        context={},
        terminology_version_id="",
        index_version="transcript-layers-v1",
        created_at="2026-08-16T14:42:00Z",
    )

    assert pre.correction_pass_count == 1
    assert post.correction_pass_count == 2
    assert replay.status == "unchanged"
    assert replay.normalized_generation_id == post.normalized_generation_id
    assert ledger.load_normalized_generation(post.normalized_generation_id)[
        "predecessor_generation_id"
    ] == pre.normalized_generation_id

    first = ledger.record_identity_cascade(
        normalized_generation_id=post.normalized_generation_id,
        processing_version="correction-v1",
        created_at="2026-08-16T14:43:00Z",
    )
    second = ledger.record_identity_cascade(
        normalized_generation_id=post.normalized_generation_id,
        processing_version="correction-v1",
        created_at="2026-08-16T14:44:00Z",
    )

    assert first.cascade_ordinal == 1
    assert first.outcome == "identity_requeue_required"
    assert second.cascade_ordinal == 2
    assert second.outcome == "manual_resolution_required"
    with pytest.raises(ValueError, match="manual resolution"):
        ledger.record_identity_cascade(
            normalized_generation_id=post.normalized_generation_id,
            processing_version="correction-v1",
            created_at="2026-08-16T14:45:00Z",
        )


def test_accepted_corrections_use_scope_precedence_and_preserve_conflicts(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)

    def raw_generation(suffix: str, source_hash: str):
        text = "CISO"
        return ledger.record_raw_transcript(
            conversation_id=f"00000000-0000-4000-8000-0000000002{suffix}",
            recording_id=f"00000000-0000-4000-8000-0000000003{suffix}",
            source_artifact_sha256=source_hash * 64,
            transcript_text=text,
            utterances=(
                {
                    "speaker": "SPEAKER_1",
                    "start_ms": 0,
                    "end_ms": 500,
                    "text": text,
                },
            ),
            captured_at="2026-08-15T22:00:00Z",
            created_at="2026-08-16T14:50:00Z",
        )

    def accepted(
        raw_generation_id: str,
        replacement: str,
        scope_type: str,
        scope_id: str,
        key: str,
    ) -> str:
        proposal = ledger.propose_correction(
            raw_generation_id=raw_generation_id,
            span_start=0,
            span_end=4,
            replacement_text=replacement,
            correction_kind="asr_confusion",
            terminology_entry_id="",
            scope={"type": scope_type, "id": scope_id},
            evidence_ids=(f"evidence-{key}",),
            confidence=0.95,
            correction_pass="pre_identity",
            processing_version="correction-v1",
            cascade_count=0,
            created_at="2026-08-16T14:51:00Z",
        )
        ledger.decide_correction(
            proposal_id=proposal.proposal_id,
            action="accept",
            reviewer="reviewer:test",
            method="fixture_review",
            decided_at="2026-08-16T14:52:00Z",
            idempotency_key=f"accept-{key}",
        )
        return proposal.proposal_id

    scoped_raw = raw_generation("41", "e")
    accepted(
        scoped_raw.raw_generation_id,
        "Chief Information Security Officer",
        "global",
        "global",
        "global",
    )
    domain_id = accepted(
        scoped_raw.raw_generation_id,
        "SESO",
        "domain",
        "chemistry",
        "domain",
    )

    normalized = ledger.normalize(
        raw_generation_id=scoped_raw.raw_generation_id,
        processing_version="correction-v1",
        correction_pass="pre_identity",
        context={"domain": "chemistry"},
        terminology_version_id="",
        index_version="transcript-layers-v1",
        created_at="2026-08-16T14:53:00Z",
    )

    assert normalized.normalized_text == "SESO"
    assert normalized.accepted_correction_ids == (domain_id,)

    conflict_raw = raw_generation("42", "f")
    accepted(
        conflict_raw.raw_generation_id,
        "SESO batch A",
        "domain",
        "chemistry",
        "conflict-a",
    )
    accepted(
        conflict_raw.raw_generation_id,
        "SESO batch B",
        "domain",
        "chemistry",
        "conflict-b",
    )
    with pytest.raises(ValueError, match="Equal-scope accepted"):
        ledger.normalize(
            raw_generation_id=conflict_raw.raw_generation_id,
            processing_version="correction-v1",
            correction_pass="pre_identity",
            context={"domain": "chemistry"},
            terminology_version_id="",
            index_version="transcript-layers-v1",
            created_at="2026-08-16T14:54:00Z",
        )


def test_correction_decisions_require_explicit_supersession_and_remain_immutable(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)
    raw = ledger.record_raw_transcript(
        conversation_id="00000000-0000-4000-8000-000000000251",
        recording_id="00000000-0000-4000-8000-000000000252",
        source_artifact_sha256="1" * 64,
        transcript_text="CISO",
        utterances=(
            {
                "speaker": "SPEAKER_1",
                "start_ms": 0,
                "end_ms": 500,
                "text": "CISO",
            },
        ),
        captured_at="2026-08-15T23:00:00Z",
        created_at="2026-08-16T15:00:00Z",
    )
    proposal = ledger.propose_correction(
        raw_generation_id=raw.raw_generation_id,
        span_start=0,
        span_end=4,
        replacement_text="SESO",
        correction_kind="asr_confusion",
        terminology_entry_id="",
        scope={"type": "domain", "id": "chemistry"},
        evidence_ids=("evidence-supersession",),
        confidence=0.97,
        correction_pass="pre_identity",
        processing_version="correction-v1",
        cascade_count=0,
        created_at="2026-08-16T15:01:00Z",
    )
    accepted = ledger.decide_correction(
        proposal_id=proposal.proposal_id,
        action="accept",
        reviewer="reviewer:test",
        method="fixture_review",
        decided_at="2026-08-16T15:02:00Z",
        idempotency_key="accept-supersession",
    )
    with pytest.raises(ValueError, match="must supersede"):
        ledger.decide_correction(
            proposal_id=proposal.proposal_id,
            action="reject",
            reviewer="reviewer:test",
            method="fixture_review",
            decided_at="2026-08-16T15:03:00Z",
            idempotency_key="reject-without-supersession",
        )
    rejected = ledger.decide_correction(
        proposal_id=proposal.proposal_id,
        action="reject",
        reviewer="reviewer:test",
        method="fixture_review",
        decided_at="2026-08-16T15:01:30Z",
        idempotency_key="reject-with-supersession",
        supersedes_decision_id=accepted.decision_id,
    )

    history = ledger.correction_decision_history(proposal.proposal_id)
    normalized = ledger.normalize(
        raw_generation_id=raw.raw_generation_id,
        processing_version="correction-v1",
        correction_pass="pre_identity",
        context={"domain": "chemistry"},
        terminology_version_id="",
        index_version="transcript-layers-v1",
        created_at="2026-08-16T15:05:00Z",
    )

    history_by_id = {item["id"]: item for item in history}
    assert {item["action"] for item in history} == {"accept", "reject"}
    assert history_by_id[rejected.decision_id]["supersedes_decision_id"] == (
        accepted.decision_id
    )
    assert normalized.accepted_correction_ids == ()
    assert normalized.normalized_text == "CISO"
    with sqlite3.connect(tmp_path / "transcripts.sqlite3") as con:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            con.execute(
                """
                UPDATE knowledge_transcript_correction_decisions
                SET reviewer = 'tampered'
                """
            )


def test_plan0072_a2_redacted_fixtures_replay_with_exact_lineage(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)
    terms = _a2_fixture("terminology-registry.json")
    replay = _a2_fixture("transcript-correction-replay.json")
    semantic = _a2_fixture("semantic-map.json")
    entries = tuple(
        TerminologyEntrySpec(
            entry_id=str(entry["entry_id"]),
            canonical_term=str(entry["canonical_term"]),
            expansion=str(entry["expansion"]),
            definition=str(entry["definition"]),
            aliases=tuple(entry["aliases"]),
            asr_confusions=tuple(entry["asr_confusions"]),
            pronunciation_hints=tuple(entry["pronunciation_hints"]),
            scope_type=str(entry["scope_type"]),
            scope_id=str(entry["scope_id"]),
            status=str(entry["status"]),
            source_observation_ids=tuple(entry["source_observation_ids"]),
            valid_from=str(entry["valid_from"]),
        )
        for entry in terms["entries"]
    )
    term_version = ledger.register_terminology(
        version=str(terms["version"]),
        entries=entries,
        status=str(terms["status"]),
        created_at=str(terms["created_at"]),
        metadata=terms["metadata"],
    )
    raw_input = replay["raw_transcript"]
    raw = ledger.record_raw_transcript(**raw_input)
    proposal_input = dict(replay["correction_proposal"])
    proposal_input["raw_generation_id"] = raw.raw_generation_id
    proposal = ledger.propose_correction(**proposal_input)
    decision_input = dict(replay["review_decision"])
    decision_input["proposal_id"] = proposal.proposal_id
    ledger.decide_correction(**decision_input)
    normalization_input = dict(replay["normalization"])
    normalization_input["raw_generation_id"] = raw.raw_generation_id
    normalization_input["terminology_version_id"] = (
        term_version.terminology_version_id
    )
    normalized = ledger.normalize(**normalization_input)
    sections = semantic["sections"]
    for section in sections.values():
        for claim in section:
            for lineage in claim["raw_lineage"]:
                lineage["raw_generation_id"] = raw.raw_generation_id
    semantic_receipt = ledger.record_semantic_map(
        normalized_generation_id=normalized.normalized_generation_id,
        sections=sections,
        created_at=str(semantic["created_at"]),
    )

    expected = replay["expected"]
    assert raw.transcript_sha256 == expected["raw_transcript_sha256"]
    assert proposal.original_text == expected["original_text"]
    assert proposal.raw_span_sha256 == expected["raw_span_sha256"]
    assert normalized.normalized_text == expected["normalized_text"]
    assert (
        normalized.normalized_transcript_sha256
        == expected["normalized_transcript_sha256"]
    )
    assert semantic_receipt.claim_count == 1
    assert ledger.load_raw_generation(raw.raw_generation_id)["transcript_text"] == (
        raw_input["transcript_text"]
    )
    assert ledger.load_reindex_receipt(normalized.normalized_generation_id)[
        "indexed_layer_count"
    ] == 2
