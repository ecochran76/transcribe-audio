"""Immutable terminology and transcript-correction generations for Plan 0072."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import NAMESPACE_URL, uuid5

import transcript_store
from identity_learning_contracts import ARTIFACT_SCHEMAS, validate_artifact


SCOPE_PRECEDENCE = (
    "conversation",
    "project_matter",
    "organization",
    "domain",
    "global",
)
SEMANTIC_MAP_SCHEMA = "transcribe-audio.transcript-semantic-map.v1"


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _stable_id(prefix: str, *parts: str) -> str:
    return f"{prefix}-{uuid5(NAMESPACE_URL, chr(31).join(parts))}"


def _text(value: object) -> str:
    return str(value or "").strip()


def _normalized_term(value: object) -> str:
    return " ".join(_text(value).casefold().split())


@dataclass(frozen=True)
class TerminologyEntrySpec:
    entry_id: str
    canonical_term: str
    expansion: str
    definition: str
    aliases: tuple[str, ...]
    asr_confusions: tuple[str, ...]
    pronunciation_hints: tuple[str, ...]
    scope_type: str
    scope_id: str
    status: str
    source_observation_ids: tuple[str, ...] = ()
    valid_from: str = ""
    valid_to: str = ""
    supersedes_entry_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TerminologyRegistrationReceipt:
    terminology_version_id: str
    version: str
    content_hash: str
    entry_count: int
    status: str


@dataclass(frozen=True)
class TerminologyResolution:
    status: str
    observed_text: str
    canonical_term: str
    scope_type: str
    match_kind: str
    candidate_entry_ids: tuple[str, ...]
    reason_code: str
    terminology_version_id: str


@dataclass(frozen=True)
class TerminologyHint:
    entry_id: str
    canonical_term: str
    aliases: tuple[str, ...]
    asr_confusions: tuple[str, ...]
    pronunciation_hints: tuple[str, ...]
    scope_type: str
    scope_id: str


@dataclass(frozen=True)
class TerminologyHintBundle:
    terminology_version_id: str
    version: str
    content_hash: str
    hints: tuple[TerminologyHint, ...]


@dataclass(frozen=True)
class RawTranscriptReceipt:
    raw_generation_id: str
    transcript_sha256: str
    diarization_sha256: str
    status: str


@dataclass(frozen=True)
class CorrectionProposalReceipt:
    proposal_id: str
    raw_span_sha256: str
    original_text: str
    status: str


@dataclass(frozen=True)
class CorrectionDecisionReceipt:
    decision_id: str
    proposal_id: str
    action: str
    status: str


@dataclass(frozen=True)
class NormalizationReceipt:
    normalized_generation_id: str
    normalized_text: str
    normalized_transcript_sha256: str
    accepted_correction_ids: tuple[str, ...]
    correction_pass_count: int
    index_version: str
    status: str


@dataclass(frozen=True)
class TranscriptLayerSearchResult:
    generation_id: str
    conversation_id: str
    recording_id: str
    layer: str
    snippet: str
    rank: float


@dataclass(frozen=True)
class SemanticMapReceipt:
    semantic_map_id: str
    normalized_generation_id: str
    claim_count: int
    content_hash: str
    status: str


@dataclass(frozen=True)
class IdentityCascadeReceipt:
    cascade_id: str
    cascade_ordinal: int
    outcome: str
    normalized_generation_id: str
    status: str


class TranscriptCorrectionLedger:
    """Expose scoped terminology and correction replay behind one interface."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = transcript_store.store_dir(root)
        self._require_v5()

    def _require_v5(self) -> None:
        with transcript_store.connect(self.root) as con:
            row = con.execute(
                """
                SELECT schema_version, dirty
                FROM knowledge_store_state
                WHERE singleton = 1
                """
            ).fetchone()
        if row is None or int(row["schema_version"]) < 5 or bool(row["dirty"]):
            raise RuntimeError(
                "Transcript correction ledger requires knowledge schema v5."
            )

    def register_terminology(
        self,
        *,
        version: str,
        entries: Sequence[TerminologyEntrySpec],
        status: str,
        created_at: str,
        predecessor_version_id: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> TerminologyRegistrationReceipt:
        version = _text(version)
        created_at = _text(created_at)
        if not version or not entries or not created_at:
            raise ValueError(
                "Terminology registration requires version, entries, and created_at."
            )
        if status not in {"draft", "reviewed", "superseded"}:
            raise ValueError("Terminology version status is invalid.")
        seen_ids: set[str] = set()
        normalized_entries: list[dict[str, Any]] = []
        for entry in entries:
            self._validate_terminology_entry(entry, version, created_at)
            if entry.entry_id in seen_ids:
                raise ValueError("Terminology entry IDs must be unique.")
            seen_ids.add(entry.entry_id)
            normalized_entries.append(asdict(entry))
        core = {
            "version": version,
            "predecessor_version_id": _text(predecessor_version_id),
            "status": status,
            "entries": normalized_entries,
            "metadata": dict(metadata or {}),
        }
        content_hash = _canonical_hash(core)
        version_id = _stable_id(
            "terminology-version",
            version,
            content_hash,
        )
        with transcript_store.connect(self.root) as con:
            existing = con.execute(
                """
                SELECT id, content_hash
                FROM knowledge_terminology_versions
                WHERE version = ?
                """,
                (version,),
            ).fetchone()
            if existing is not None:
                if str(existing["content_hash"]) != content_hash:
                    raise ValueError(
                        "Terminology version already exists with different content."
                    )
                return TerminologyRegistrationReceipt(
                    terminology_version_id=str(existing["id"]),
                    version=version,
                    content_hash=content_hash,
                    entry_count=len(entries),
                    status="unchanged",
                )
            if predecessor_version_id:
                predecessor = con.execute(
                    "SELECT 1 FROM knowledge_terminology_versions WHERE id = ?",
                    (predecessor_version_id,),
                ).fetchone()
                if predecessor is None:
                    raise ValueError("Terminology predecessor version is unknown.")
            con.execute("BEGIN IMMEDIATE")
            try:
                con.execute(
                    """
                    INSERT INTO knowledge_terminology_versions (
                        id, version, predecessor_version_id, status,
                        content_hash, metadata_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        version_id,
                        version,
                        _text(predecessor_version_id) or None,
                        status,
                        content_hash,
                        _canonical_json(dict(metadata or {})),
                        created_at,
                    ),
                )
                for entry in entries:
                    entry_payload = asdict(entry)
                    entry_hash = _canonical_hash(
                        {
                            "terminology_version_id": version_id,
                            **entry_payload,
                        }
                    )
                    con.execute(
                        """
                        INSERT INTO knowledge_terminology_entries (
                            id, terminology_version_id, canonical_term,
                            expansion, definition, aliases_json,
                            asr_confusions_json, pronunciation_hints_json,
                            scope_type, scope_id, source_observation_ids_json,
                            valid_from, valid_to, status, content_hash,
                            metadata_json, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            entry.entry_id,
                            version_id,
                            entry.canonical_term,
                            entry.expansion,
                            entry.definition,
                            _canonical_json(list(entry.aliases)),
                            _canonical_json(list(entry.asr_confusions)),
                            _canonical_json(list(entry.pronunciation_hints)),
                            entry.scope_type,
                            entry.scope_id,
                            _canonical_json(list(entry.source_observation_ids)),
                            entry.valid_from,
                            entry.valid_to,
                            entry.status,
                            entry_hash,
                            _canonical_json(dict(entry.metadata)),
                            created_at,
                        ),
                    )
                con.commit()
            except Exception:
                con.rollback()
                raise
        return TerminologyRegistrationReceipt(
            terminology_version_id=version_id,
            version=version,
            content_hash=content_hash,
            entry_count=len(entries),
            status="inserted",
        )

    def resolve_terminology(
        self,
        observed_text: str,
        *,
        terminology_version_id: str,
        context: Mapping[str, str],
    ) -> TerminologyResolution:
        observed = _normalized_term(observed_text)
        if not observed:
            raise ValueError("Terminology resolution requires observed text.")
        with transcript_store.connect(self.root) as con:
            version = con.execute(
                """
                SELECT status
                FROM knowledge_terminology_versions
                WHERE id = ?
                """,
                (terminology_version_id,),
            ).fetchone()
            if version is None:
                raise ValueError("Terminology version is unknown.")
            if str(version["status"]) != "reviewed":
                raise ValueError(
                    "Only reviewed terminology versions may resolve terms."
                )
            rows = con.execute(
                """
                SELECT *
                FROM knowledge_terminology_entries
                WHERE terminology_version_id = ? AND status = 'reviewed'
                ORDER BY id
                """,
                (terminology_version_id,),
            ).fetchall()
        matches: list[tuple[int, str, str, str, str]] = []
        for row in rows:
            scope_type = str(row["scope_type"])
            scope_id = str(row["scope_id"])
            if not self._scope_applies(scope_type, scope_id, context):
                continue
            canonical = str(row["canonical_term"])
            aliases = tuple(json.loads(str(row["aliases_json"])))
            confusions = tuple(json.loads(str(row["asr_confusions_json"])))
            match_kind = ""
            if observed == _normalized_term(canonical):
                match_kind = "canonical"
            elif observed in {_normalized_term(value) for value in aliases}:
                match_kind = "alias"
            elif observed in {_normalized_term(value) for value in confusions}:
                match_kind = "asr_confusion"
            if match_kind:
                matches.append(
                    (
                        SCOPE_PRECEDENCE.index(scope_type),
                        str(row["id"]),
                        canonical,
                        scope_type,
                        match_kind,
                    )
                )
        if not matches:
            return TerminologyResolution(
                status="unmatched",
                observed_text=observed_text,
                canonical_term="",
                scope_type="",
                match_kind="",
                candidate_entry_ids=(),
                reason_code="no_scoped_match",
                terminology_version_id=terminology_version_id,
            )
        best_rank = min(item[0] for item in matches)
        best = [item for item in matches if item[0] == best_rank]
        candidate_ids = tuple(item[1] for item in best)
        canonical_values = {_normalized_term(item[2]) for item in best}
        if len(canonical_values) > 1:
            return TerminologyResolution(
                status="review_required",
                observed_text=observed_text,
                canonical_term="",
                scope_type=best[0][3],
                match_kind="",
                candidate_entry_ids=candidate_ids,
                reason_code="equal_scope_conflict",
                terminology_version_id=terminology_version_id,
            )
        return TerminologyResolution(
            status="resolved",
            observed_text=observed_text,
            canonical_term=best[0][2],
            scope_type=best[0][3],
            match_kind=best[0][4],
            candidate_entry_ids=candidate_ids,
            reason_code="scoped_exact_match",
            terminology_version_id=terminology_version_id,
        )

    def terminology_hints(
        self,
        *,
        terminology_version_id: str,
        context: Mapping[str, str],
    ) -> TerminologyHintBundle:
        """Return reviewed, applicable terms without activating a provider."""
        with transcript_store.connect(self.root) as con:
            version = con.execute(
                """
                SELECT version, status, content_hash
                FROM knowledge_terminology_versions
                WHERE id = ?
                """,
                (terminology_version_id,),
            ).fetchone()
            if version is None:
                raise ValueError("Terminology version is unknown.")
            if str(version["status"]) != "reviewed":
                raise ValueError(
                    "Only reviewed terminology versions may produce hints."
                )
            rows = con.execute(
                """
                SELECT *
                FROM knowledge_terminology_entries
                WHERE terminology_version_id = ? AND status = 'reviewed'
                ORDER BY id
                """,
                (terminology_version_id,),
            ).fetchall()
        applicable = [
            row
            for row in rows
            if self._scope_applies(
                str(row["scope_type"]), str(row["scope_id"]), context
            )
        ]
        applicable.sort(
            key=lambda row: (
                SCOPE_PRECEDENCE.index(str(row["scope_type"])),
                str(row["id"]),
            )
        )
        hints = tuple(
            TerminologyHint(
                entry_id=str(row["id"]),
                canonical_term=str(row["canonical_term"]),
                aliases=tuple(json.loads(str(row["aliases_json"]))),
                asr_confusions=tuple(
                    json.loads(str(row["asr_confusions_json"]))
                ),
                pronunciation_hints=tuple(
                    json.loads(str(row["pronunciation_hints_json"]))
                ),
                scope_type=str(row["scope_type"]),
                scope_id=str(row["scope_id"]),
            )
            for row in applicable
        )
        return TerminologyHintBundle(
            terminology_version_id=terminology_version_id,
            version=str(version["version"]),
            content_hash=str(version["content_hash"]),
            hints=hints,
        )

    def record_raw_transcript(
        self,
        *,
        conversation_id: str,
        recording_id: str,
        source_artifact_sha256: str,
        transcript_text: str,
        utterances: Sequence[Mapping[str, Any]],
        captured_at: str,
        created_at: str,
    ) -> RawTranscriptReceipt:
        conversation_id = _text(conversation_id)
        recording_id = _text(recording_id)
        transcript_text = str(transcript_text)
        created_at = _text(created_at)
        self._require_sha256(source_artifact_sha256, "source_artifact_sha256")
        if not all((conversation_id, recording_id, transcript_text, created_at)):
            raise ValueError(
                "Raw transcript requires conversation, recording, text, and created_at."
            )
        prepared_utterances: list[dict[str, Any]] = []
        for utterance in utterances:
            if not isinstance(utterance, Mapping):
                raise ValueError("Raw transcript utterances must be objects.")
            speaker = _text(utterance.get("speaker"))
            text = str(utterance.get("text") or "")
            start_ms = utterance.get("start_ms")
            end_ms = utterance.get("end_ms")
            if (
                not speaker
                or not text
                or not isinstance(start_ms, int)
                or not isinstance(end_ms, int)
                or start_ms < 0
                or end_ms <= start_ms
            ):
                raise ValueError("Raw transcript utterance diarization is invalid.")
            prepared_utterances.append(
                {
                    "speaker": speaker,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "text": text,
                }
            )
        if not prepared_utterances:
            raise ValueError("Raw transcript requires diarized utterances.")
        transcript_sha256 = hashlib.sha256(
            transcript_text.encode("utf-8")
        ).hexdigest()
        diarization_sha256 = _canonical_hash(prepared_utterances)
        core = {
            "conversation_id": conversation_id,
            "recording_id": recording_id,
            "source_artifact_sha256": source_artifact_sha256,
            "transcript_sha256": transcript_sha256,
            "diarization_sha256": diarization_sha256,
            "transcript_text": transcript_text,
            "utterances": prepared_utterances,
            "captured_at": _text(captured_at),
        }
        content_hash = _canonical_hash(core)
        generation_id = _stable_id(
            "raw-transcript",
            conversation_id,
            recording_id,
            source_artifact_sha256,
            content_hash,
        )
        with transcript_store.connect(self.root) as con:
            existing = con.execute(
                """
                SELECT id, content_hash
                FROM knowledge_raw_transcript_generations
                WHERE conversation_id = ? AND recording_id = ?
                  AND source_artifact_sha256 = ?
                """,
                (conversation_id, recording_id, source_artifact_sha256),
            ).fetchone()
            if existing is not None:
                if str(existing["content_hash"]) != content_hash:
                    raise ValueError(
                        "Raw transcript source identity already exists with "
                        "different content."
                    )
                return RawTranscriptReceipt(
                    str(existing["id"]),
                    transcript_sha256,
                    diarization_sha256,
                    "unchanged",
                )
            con.execute(
                """
                INSERT INTO knowledge_raw_transcript_generations (
                    id, conversation_id, recording_id,
                    source_artifact_sha256, transcript_sha256,
                    diarization_sha256, transcript_text, utterances_json,
                    captured_at, content_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    generation_id,
                    conversation_id,
                    recording_id,
                    source_artifact_sha256,
                    transcript_sha256,
                    diarization_sha256,
                    transcript_text,
                    _canonical_json(prepared_utterances),
                    _text(captured_at),
                    content_hash,
                    created_at,
                ),
            )
            con.commit()
        return RawTranscriptReceipt(
            generation_id,
            transcript_sha256,
            diarization_sha256,
            "inserted",
        )

    def propose_correction(
        self,
        *,
        raw_generation_id: str,
        span_start: int,
        span_end: int,
        replacement_text: str,
        correction_kind: str,
        terminology_entry_id: str,
        scope: Mapping[str, str],
        evidence_ids: Sequence[str],
        confidence: float | None,
        correction_pass: str,
        processing_version: str,
        cascade_count: int,
        created_at: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> CorrectionProposalReceipt:
        raw = self.load_raw_generation(raw_generation_id)
        raw_text = str(raw["transcript_text"])
        if (
            not isinstance(span_start, int)
            or not isinstance(span_end, int)
            or span_start < 0
            or span_end <= span_start
            or span_end > len(raw_text)
        ):
            raise ValueError("Transcript correction span is out of bounds.")
        replacement_text = str(replacement_text)
        if not replacement_text or not _text(correction_kind):
            raise ValueError(
                "Transcript correction requires kind and replacement text."
            )
        scope_type = _text(scope.get("type"))
        scope_id = _text(scope.get("id"))
        if scope_type not in SCOPE_PRECEDENCE or not scope_id:
            raise ValueError("Transcript correction scope is invalid.")
        if correction_pass not in {"pre_identity", "post_identity"}:
            raise ValueError("Transcript correction pass is invalid.")
        if cascade_count not in {0, 1}:
            raise ValueError("Transcript correction violates the one-cascade limit.")
        if not _text(processing_version) or not _text(created_at):
            raise ValueError("Transcript correction requires version and created_at.")
        if confidence is not None and not 0.0 <= float(confidence) <= 1.0:
            raise ValueError("Transcript correction confidence must be 0..1.")
        evidence = tuple(dict.fromkeys(_text(item) for item in evidence_ids))
        if not evidence or any(not item for item in evidence):
            raise ValueError("Transcript correction requires evidence IDs.")
        if terminology_entry_id:
            with transcript_store.connect(self.root) as con:
                term = con.execute(
                    "SELECT 1 FROM knowledge_terminology_entries WHERE id = ?",
                    (terminology_entry_id,),
                ).fetchone()
            if term is None:
                raise ValueError("Transcript correction terminology entry is unknown.")
        original_text = raw_text[span_start:span_end]
        raw_span_sha256 = hashlib.sha256(
            original_text.encode("utf-8")
        ).hexdigest()
        core = {
            "raw_generation_id": raw_generation_id,
            "conversation_id": raw["conversation_id"],
            "recording_id": raw["recording_id"],
            "raw_transcript_sha256": raw["transcript_sha256"],
            "span_start": span_start,
            "span_end": span_end,
            "raw_span_sha256": raw_span_sha256,
            "original_text": original_text,
            "replacement_text": replacement_text,
            "correction_kind": _text(correction_kind),
            "terminology_entry_id": _text(terminology_entry_id),
            "scope_type": scope_type,
            "scope_id": scope_id,
            "evidence_ids": evidence,
            "confidence": confidence,
            "review_state": "proposed",
            "correction_pass": correction_pass,
            "processing_version": _text(processing_version),
            "cascade_count": cascade_count,
            "metadata": dict(metadata or {}),
        }
        content_hash = _canonical_hash(core)
        proposal_id = _stable_id(
            "transcript-correction",
            raw_generation_id,
            str(span_start),
            str(span_end),
            content_hash,
        )
        validate_artifact(
            "transcript_correction_proposal",
            {
                "schema_version": ARTIFACT_SCHEMAS[
                    "transcript_correction_proposal"
                ],
                "proposal_id": proposal_id,
                "conversation_id": raw["conversation_id"],
                "recording_id": raw["recording_id"],
                "raw_transcript_sha256": raw["transcript_sha256"],
                "raw_span": {
                    "start": span_start,
                    "end": span_end,
                    "text_sha256": raw_span_sha256,
                },
                "replacement_text": replacement_text,
                "correction_kind": _text(correction_kind),
                "terminology_entry_id": _text(terminology_entry_id),
                "scope": {"type": scope_type, "id": scope_id},
                "evidence_ids": list(evidence),
                "review_state": "proposed",
                "correction_pass": correction_pass,
                "processing_version": _text(processing_version),
                "cascade_count": cascade_count,
                "created_at": _text(created_at),
            },
        )
        with transcript_store.connect(self.root) as con:
            existing = con.execute(
                """
                SELECT content_hash
                FROM knowledge_transcript_correction_proposals
                WHERE id = ?
                """,
                (proposal_id,),
            ).fetchone()
            if existing is not None:
                if str(existing["content_hash"]) != content_hash:
                    raise ValueError("Transcript correction proposal hash drifted.")
                return CorrectionProposalReceipt(
                    proposal_id,
                    raw_span_sha256,
                    original_text,
                    "unchanged",
                )
            con.execute(
                """
                INSERT INTO knowledge_transcript_correction_proposals (
                    id, raw_generation_id, conversation_id, recording_id,
                    raw_transcript_sha256, span_start, span_end,
                    raw_span_sha256, original_text, replacement_text,
                    correction_kind, terminology_entry_id, scope_type,
                    scope_id, evidence_ids_json, confidence, review_state,
                    correction_pass, processing_version, cascade_count,
                    content_hash, metadata_json, created_at
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?
                )
                """,
                (
                    proposal_id,
                    raw_generation_id,
                    raw["conversation_id"],
                    raw["recording_id"],
                    raw["transcript_sha256"],
                    span_start,
                    span_end,
                    raw_span_sha256,
                    original_text,
                    replacement_text,
                    _text(correction_kind),
                    _text(terminology_entry_id) or None,
                    scope_type,
                    scope_id,
                    _canonical_json(list(evidence)),
                    confidence,
                    "proposed",
                    correction_pass,
                    _text(processing_version),
                    cascade_count,
                    content_hash,
                    _canonical_json(dict(metadata or {})),
                    _text(created_at),
                ),
            )
            con.commit()
        return CorrectionProposalReceipt(
            proposal_id,
            raw_span_sha256,
            original_text,
            "inserted",
        )

    def decide_correction(
        self,
        *,
        proposal_id: str,
        action: str,
        reviewer: str,
        method: str,
        decided_at: str,
        idempotency_key: str,
        supersedes_decision_id: str = "",
        comment: str = "",
    ) -> CorrectionDecisionReceipt:
        if action not in {"accept", "reject", "defer", "supersede"}:
            raise ValueError("Transcript correction decision action is invalid.")
        if not all(
            map(_text, (proposal_id, reviewer, method, decided_at, idempotency_key))
        ):
            raise ValueError("Transcript correction decision is incomplete.")
        with transcript_store.connect(self.root) as con:
            proposal = con.execute(
                """
                SELECT 1
                FROM knowledge_transcript_correction_proposals
                WHERE id = ?
                """,
                (proposal_id,),
            ).fetchone()
            if proposal is None:
                raise ValueError("Transcript correction proposal is unknown.")
            core = {
                "proposal_id": proposal_id,
                "action": action,
                "reviewer": _text(reviewer),
                "method": _text(method),
                "decided_at": _text(decided_at),
                "supersedes_decision_id": _text(supersedes_decision_id),
                "comment": str(comment),
                "idempotency_key": _text(idempotency_key),
            }
            content_hash = _canonical_hash(core)
            decision_id = _stable_id(
                "correction-decision",
                _text(idempotency_key),
            )
            existing = con.execute(
                """
                SELECT id, content_hash
                FROM knowledge_transcript_correction_decisions
                WHERE idempotency_key = ?
                """,
                (_text(idempotency_key),),
            ).fetchone()
            if existing is not None:
                if str(existing["content_hash"]) != content_hash:
                    raise ValueError(
                        "Correction decision idempotency key was reused with "
                        "different content."
                    )
                return CorrectionDecisionReceipt(
                    str(existing["id"]), proposal_id, action, "unchanged"
                )
            current_rows = con.execute(
                """
                SELECT decision.id
                FROM knowledge_transcript_correction_decisions AS decision
                WHERE decision.proposal_id = ?
                  AND NOT EXISTS (
                      SELECT 1
                      FROM knowledge_transcript_correction_decisions AS successor
                      WHERE successor.supersedes_decision_id = decision.id
                  )
                ORDER BY decision.decided_at, decision.id
                """,
                (proposal_id,),
            ).fetchall()
            if len(current_rows) > 1:
                raise RuntimeError(
                    "Transcript correction decision history has multiple heads."
                )
            current_decision_id = (
                str(current_rows[0]["id"]) if current_rows else ""
            )
            if current_decision_id and not supersedes_decision_id:
                raise ValueError(
                    "New correction decision must supersede the current decision."
                )
            if supersedes_decision_id != current_decision_id:
                if supersedes_decision_id:
                    raise ValueError(
                        "Superseded correction decision is not the current decision."
                    )
                raise ValueError(
                    "Correction decision cannot supersede an empty history."
                )
            con.execute(
                """
                INSERT INTO knowledge_transcript_correction_decisions (
                    id, proposal_id, action, reviewer, method, decided_at,
                    supersedes_decision_id, comment, idempotency_key,
                    content_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    decision_id,
                    proposal_id,
                    action,
                    _text(reviewer),
                    _text(method),
                    _text(decided_at),
                    _text(supersedes_decision_id) or None,
                    str(comment),
                    _text(idempotency_key),
                    content_hash,
                    _text(decided_at),
                ),
            )
            con.commit()
        return CorrectionDecisionReceipt(
            decision_id, proposal_id, action, "inserted"
        )

    def correction_decision_history(
        self,
        proposal_id: str,
    ) -> tuple[dict[str, Any], ...]:
        with transcript_store.connect(self.root) as con:
            rows = con.execute(
                """
                SELECT *
                FROM knowledge_transcript_correction_decisions
                WHERE proposal_id = ?
                ORDER BY decided_at, id
                """,
                (proposal_id,),
            ).fetchall()
        return tuple(dict(row) for row in rows)

    def normalize(
        self,
        *,
        raw_generation_id: str,
        processing_version: str,
        correction_pass: str,
        context: Mapping[str, str],
        terminology_version_id: str,
        index_version: str,
        created_at: str,
    ) -> NormalizationReceipt:
        if correction_pass not in {"pre_identity", "post_identity"}:
            raise ValueError("Transcript correction pass is invalid.")
        if not all(map(_text, (processing_version, index_version, created_at))):
            raise ValueError("Normalization requires processing, index, and time.")
        raw = self.load_raw_generation(raw_generation_id)
        with transcript_store.connect(self.root) as con:
            existing_run = con.execute(
                """
                SELECT output_generation_id
                FROM knowledge_transcript_correction_runs
                WHERE conversation_id = ? AND recording_id = ?
                  AND processing_version = ? AND correction_pass = ?
                """,
                (
                    raw["conversation_id"],
                    raw["recording_id"],
                    _text(processing_version),
                    correction_pass,
                ),
            ).fetchone()
            if existing_run is not None:
                loaded = self.load_normalized_generation(
                    str(existing_run["output_generation_id"])
                )
                return self._normalization_receipt(loaded, "unchanged")
            if terminology_version_id:
                version = con.execute(
                    """
                    SELECT status FROM knowledge_terminology_versions WHERE id = ?
                    """,
                    (terminology_version_id,),
                ).fetchone()
                if version is None or str(version["status"]) != "reviewed":
                    raise ValueError(
                        "Normalization requires a reviewed terminology version."
                    )
            proposals = con.execute(
                """
                SELECT *
                FROM knowledge_transcript_correction_proposals
                WHERE raw_generation_id = ? AND processing_version = ?
                  AND correction_pass = ?
                ORDER BY span_start, span_end, id
                """,
                (raw_generation_id, _text(processing_version), correction_pass),
            ).fetchall()
            decisions = con.execute(
                """
                SELECT d.*
                FROM knowledge_transcript_correction_decisions d
                JOIN knowledge_transcript_correction_proposals p
                  ON p.id = d.proposal_id
                WHERE p.raw_generation_id = ? AND p.processing_version = ?
                  AND p.correction_pass = ?
                  AND NOT EXISTS (
                      SELECT 1
                      FROM knowledge_transcript_correction_decisions successor
                      WHERE successor.supersedes_decision_id = d.id
                  )
                ORDER BY d.proposal_id, d.id
                """,
                (raw_generation_id, _text(processing_version), correction_pass),
            ).fetchall()
            prior_runs = con.execute(
                """
                SELECT COUNT(*) AS count
                FROM knowledge_transcript_correction_runs
                WHERE conversation_id = ? AND recording_id = ?
                  AND processing_version = ?
                """,
                (
                    raw["conversation_id"],
                    raw["recording_id"],
                    _text(processing_version),
                ),
            ).fetchone()
            current = con.execute(
                """
                SELECT normalized_generation_id
                FROM knowledge_current_normalized_transcripts
                WHERE conversation_id = ? AND recording_id = ?
                """,
                (raw["conversation_id"], raw["recording_id"]),
            ).fetchone()
        pass_count = int(prior_runs["count"]) + 1
        if pass_count > 2:
            raise ValueError("Processing version exceeded two correction passes.")
        latest_decision: dict[str, Any] = {}
        for decision in decisions:
            latest_decision[str(decision["proposal_id"])] = decision
        accepted = [
            proposal
            for proposal in proposals
            if proposal["id"] in latest_decision
            and latest_decision[str(proposal["id"])]["action"] == "accept"
            and self._scope_applies(
                str(proposal["scope_type"]),
                str(proposal["scope_id"]),
                context,
            )
        ]
        selected = self._select_corrections(accepted)
        normalized_text, span_map = self._apply_corrections(
            str(raw["transcript_text"]),
            selected,
        )
        accepted_ids = tuple(str(row["id"]) for row in selected)
        normalized_sha256 = hashlib.sha256(
            normalized_text.encode("utf-8")
        ).hexdigest()
        predecessor_id = (
            str(current["normalized_generation_id"]) if current is not None else ""
        )
        cascade_count = max(
            (int(row["cascade_count"]) for row in selected),
            default=0,
        )
        core = {
            "conversation_id": raw["conversation_id"],
            "recording_id": raw["recording_id"],
            "raw_generation_id": raw_generation_id,
            "predecessor_generation_id": predecessor_id,
            "terminology_version_id": _text(terminology_version_id),
            "accepted_correction_ids": accepted_ids,
            "normalized_transcript_sha256": normalized_sha256,
            "raw_to_normalized_map": span_map,
            "index_version": _text(index_version),
            "correction_pass_count": pass_count,
            "identity_cascade_count": cascade_count,
            "status": "accepted",
            "processing_version": _text(processing_version),
        }
        content_hash = _canonical_hash(core)
        generation_id = _stable_id(
            "normalized-transcript",
            raw_generation_id,
            _text(processing_version),
            correction_pass,
            content_hash,
        )
        run_core = {
            "conversation_id": raw["conversation_id"],
            "recording_id": raw["recording_id"],
            "processing_version": _text(processing_version),
            "correction_pass": correction_pass,
            "raw_generation_id": raw_generation_id,
            "input_generation_id": predecessor_id,
            "output_generation_id": generation_id,
            "material_identity_change": False,
            "outcome": (
                "normalized" if accepted_ids else "no_accepted_corrections"
            ),
        }
        run_hash = _canonical_hash(run_core)
        run_id = _stable_id(
            "correction-run",
            str(raw["conversation_id"]),
            str(raw["recording_id"]),
            _text(processing_version),
            correction_pass,
        )
        reindex_core = {
            "raw_generation_id": raw_generation_id,
            "normalized_generation_id": generation_id,
            "index_version": _text(index_version),
            "raw_transcript_sha256": raw["transcript_sha256"],
            "normalized_transcript_sha256": normalized_sha256,
            "indexed_layer_count": 2,
        }
        reindex_hash = _canonical_hash(reindex_core)
        reindex_id = _stable_id("transcript-reindex", generation_id, reindex_hash)
        with transcript_store.connect(self.root) as con:
            con.execute("BEGIN IMMEDIATE")
            try:
                con.execute(
                    """
                    INSERT INTO knowledge_normalized_transcript_generations (
                        id, conversation_id, recording_id, raw_generation_id,
                        predecessor_generation_id, terminology_version_id,
                        accepted_correction_ids_json, normalized_text,
                        normalized_transcript_sha256,
                        raw_to_normalized_map_json, index_version,
                        correction_pass_count, identity_cascade_count, status,
                        processing_version, content_hash, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        generation_id,
                        raw["conversation_id"],
                        raw["recording_id"],
                        raw_generation_id,
                        predecessor_id or None,
                        _text(terminology_version_id) or None,
                        _canonical_json(list(accepted_ids)),
                        normalized_text,
                        normalized_sha256,
                        _canonical_json(span_map),
                        _text(index_version),
                        pass_count,
                        cascade_count,
                        "accepted",
                        _text(processing_version),
                        content_hash,
                        _text(created_at),
                    ),
                )
                con.execute(
                    """
                    INSERT INTO knowledge_transcript_correction_runs (
                        id, conversation_id, recording_id, processing_version,
                        correction_pass, raw_generation_id,
                        input_generation_id, output_generation_id,
                        material_identity_change, outcome, content_hash,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        raw["conversation_id"],
                        raw["recording_id"],
                        _text(processing_version),
                        correction_pass,
                        raw_generation_id,
                        predecessor_id or None,
                        generation_id,
                        0,
                        run_core["outcome"],
                        run_hash,
                        _text(created_at),
                    ),
                )
                con.execute(
                    """
                    INSERT INTO knowledge_current_normalized_transcripts (
                        conversation_id, recording_id,
                        normalized_generation_id, input_watermark, built_at
                    ) VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(conversation_id, recording_id) DO UPDATE SET
                        normalized_generation_id = excluded.normalized_generation_id,
                        input_watermark = excluded.input_watermark,
                        built_at = excluded.built_at
                    """,
                    (
                        raw["conversation_id"],
                        raw["recording_id"],
                        generation_id,
                        content_hash,
                        _text(created_at),
                    ),
                )
                con.execute(
                    """
                    DELETE FROM knowledge_transcript_layers_fts
                    WHERE conversation_id = ? AND recording_id = ?
                    """,
                    (raw["conversation_id"], raw["recording_id"]),
                )
                con.executemany(
                    """
                    INSERT INTO knowledge_transcript_layers_fts (
                        generation_id, conversation_id, recording_id, layer, text
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        (
                            raw_generation_id,
                            raw["conversation_id"],
                            raw["recording_id"],
                            "raw",
                            raw["transcript_text"],
                        ),
                        (
                            generation_id,
                            raw["conversation_id"],
                            raw["recording_id"],
                            "normalized",
                            normalized_text,
                        ),
                    ),
                )
                con.execute(
                    """
                    INSERT INTO knowledge_transcript_reindex_receipts (
                        id, raw_generation_id, normalized_generation_id,
                        index_version, raw_transcript_sha256,
                        normalized_transcript_sha256, indexed_layer_count,
                        content_hash, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        reindex_id,
                        raw_generation_id,
                        generation_id,
                        _text(index_version),
                        raw["transcript_sha256"],
                        normalized_sha256,
                        2,
                        reindex_hash,
                        _text(created_at),
                    ),
                )
                con.commit()
            except Exception:
                con.rollback()
                raise
        return NormalizationReceipt(
            normalized_generation_id=generation_id,
            normalized_text=normalized_text,
            normalized_transcript_sha256=normalized_sha256,
            accepted_correction_ids=accepted_ids,
            correction_pass_count=pass_count,
            index_version=_text(index_version),
            status="inserted",
        )

    def load_raw_generation(self, raw_generation_id: str) -> dict[str, Any]:
        with transcript_store.connect(self.root) as con:
            row = con.execute(
                """
                SELECT * FROM knowledge_raw_transcript_generations WHERE id = ?
                """,
                (raw_generation_id,),
            ).fetchone()
        if row is None:
            raise ValueError("Raw transcript generation is unknown.")
        result = dict(row)
        result["utterances"] = json.loads(str(row["utterances_json"]))
        return result

    def search_transcripts(
        self,
        query: str,
        *,
        limit: int = 50,
    ) -> tuple[TranscriptLayerSearchResult, ...]:
        if limit < 1 or limit > 100:
            raise ValueError("Transcript layer search limit must be 1..100.")
        terms = re.findall(r"[A-Za-z0-9]+", str(query))
        if not terms:
            raise ValueError("Transcript layer search requires searchable terms.")
        fts_query = " AND ".join(
            f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms
        )
        with transcript_store.connect(self.root) as con:
            rows = con.execute(
                """
                SELECT generation_id, conversation_id, recording_id, layer,
                       snippet(
                           knowledge_transcript_layers_fts,
                           4,
                           '[',
                           ']',
                           ' … ',
                           12
                       ) AS matched_snippet,
                       bm25(knowledge_transcript_layers_fts) AS rank
                FROM knowledge_transcript_layers_fts
                WHERE knowledge_transcript_layers_fts MATCH ?
                ORDER BY rank, layer, generation_id
                LIMIT ?
                """,
                (fts_query, limit),
            ).fetchall()
        return tuple(
            TranscriptLayerSearchResult(
                generation_id=str(row["generation_id"]),
                conversation_id=str(row["conversation_id"]),
                recording_id=str(row["recording_id"]),
                layer=str(row["layer"]),
                snippet=str(row["matched_snippet"]),
                rank=float(row["rank"]),
            )
            for row in rows
        )

    def record_semantic_map(
        self,
        *,
        normalized_generation_id: str,
        sections: Mapping[str, Sequence[Mapping[str, Any]]],
        created_at: str,
    ) -> SemanticMapReceipt:
        created_at = _text(created_at)
        if not created_at:
            raise ValueError("Transcript semantic map requires created_at.")
        normalized = self.load_normalized_generation(normalized_generation_id)
        raw = self.load_raw_generation(str(normalized["raw_generation_id"]))
        expected_sections = {"topics", "terms", "entities", "questions"}
        if set(sections) != expected_sections:
            raise ValueError(
                "Transcript semantic map requires only topics, terms, entities, "
                "and questions."
            )
        normalized_text = str(normalized["normalized_text"])
        raw_text = str(raw["transcript_text"])
        prepared: dict[str, list[dict[str, Any]]] = {}
        claim_count = 0
        for section_name in ("topics", "terms", "entities", "questions"):
            values = sections[section_name]
            if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
                raise ValueError("Transcript semantic map sections must be lists.")
            prepared[section_name] = []
            for value in values:
                if not isinstance(value, Mapping) or not _text(value.get("label")):
                    raise ValueError("Transcript semantic map claim requires a label.")
                extra_fields = set(value) - {
                    "label",
                    "normalized_span",
                    "raw_lineage",
                    "metadata",
                }
                if extra_fields:
                    raise ValueError(
                        "Transcript-only semantic map contains enrichment fields."
                    )
                normalized_span = self._validated_span(
                    value.get("normalized_span"),
                    normalized_text,
                    kind="normalized",
                )
                lineage = value.get("raw_lineage")
                if (
                    isinstance(lineage, (str, bytes))
                    or not isinstance(lineage, Sequence)
                    or not lineage
                ):
                    raise ValueError(
                        "Transcript semantic map claim requires raw lineage."
                    )
                prepared_lineage: list[dict[str, Any]] = []
                for raw_span_value in lineage:
                    if not isinstance(raw_span_value, Mapping):
                        raise ValueError("Transcript semantic raw lineage is invalid.")
                    if _text(raw_span_value.get("raw_generation_id")) != str(
                        raw["id"]
                    ):
                        raise ValueError(
                            "Transcript semantic raw lineage references another "
                            "generation."
                        )
                    raw_span = self._validated_span(
                        raw_span_value,
                        raw_text,
                        kind="raw lineage",
                    )
                    prepared_lineage.append(
                        {
                            "raw_generation_id": str(raw["id"]),
                            **raw_span,
                        }
                    )
                prepared[section_name].append(
                    {
                        "label": _text(value["label"]),
                        "normalized_span": normalized_span,
                        "raw_lineage": prepared_lineage,
                        **(
                            {"metadata": dict(value["metadata"])}
                            if isinstance(value.get("metadata"), Mapping)
                            else {}
                        ),
                    }
                )
                claim_count += 1
        core = {
            "normalized_generation_id": normalized_generation_id,
            "conversation_id": normalized["conversation_id"],
            "recording_id": normalized["recording_id"],
            "map_schema": SEMANTIC_MAP_SCHEMA,
            "sections": prepared,
            "transcript_only": True,
        }
        content_hash = _canonical_hash(core)
        semantic_map_id = _stable_id(
            "transcript-semantic-map",
            normalized_generation_id,
            content_hash,
        )
        with transcript_store.connect(self.root) as con:
            existing = con.execute(
                """
                SELECT content_hash
                FROM knowledge_transcript_semantic_maps
                WHERE id = ?
                """,
                (semantic_map_id,),
            ).fetchone()
            if existing is not None:
                if str(existing["content_hash"]) != content_hash:
                    raise ValueError("Transcript semantic map hash drifted.")
                return SemanticMapReceipt(
                    semantic_map_id,
                    normalized_generation_id,
                    claim_count,
                    content_hash,
                    "unchanged",
                )
            con.execute(
                """
                INSERT INTO knowledge_transcript_semantic_maps (
                    id, normalized_generation_id, conversation_id,
                    recording_id, map_schema, map_json, transcript_only,
                    content_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    semantic_map_id,
                    normalized_generation_id,
                    normalized["conversation_id"],
                    normalized["recording_id"],
                    SEMANTIC_MAP_SCHEMA,
                    _canonical_json(prepared),
                    1,
                    content_hash,
                    created_at,
                ),
            )
            con.commit()
        return SemanticMapReceipt(
            semantic_map_id,
            normalized_generation_id,
            claim_count,
            content_hash,
            "inserted",
        )

    def record_identity_cascade(
        self,
        *,
        normalized_generation_id: str,
        processing_version: str,
        created_at: str,
    ) -> IdentityCascadeReceipt:
        normalized = self.load_normalized_generation(normalized_generation_id)
        processing_version = _text(processing_version)
        created_at = _text(created_at)
        if not processing_version or not created_at:
            raise ValueError("Identity cascade requires processing version and time.")
        if str(normalized["processing_version"]) != processing_version:
            raise ValueError(
                "Identity cascade processing version does not match generation."
            )
        with transcript_store.connect(self.root) as con:
            rows = con.execute(
                """
                SELECT *
                FROM knowledge_transcript_identity_cascades
                WHERE conversation_id = ? AND recording_id = ?
                  AND processing_version = ?
                ORDER BY cascade_ordinal
                """,
                (
                    normalized["conversation_id"],
                    normalized["recording_id"],
                    processing_version,
                ),
            ).fetchall()
            if len(rows) >= 2:
                raise ValueError(
                    "Processing version already requires manual resolution."
                )
            cascade_ordinal = len(rows) + 1
            outcome = (
                "identity_requeue_required"
                if cascade_ordinal == 1
                else "manual_resolution_required"
            )
            core = {
                "conversation_id": normalized["conversation_id"],
                "recording_id": normalized["recording_id"],
                "processing_version": processing_version,
                "cascade_ordinal": cascade_ordinal,
                "triggering_generation_id": normalized_generation_id,
                "outcome": outcome,
            }
            content_hash = _canonical_hash(core)
            cascade_id = _stable_id(
                "transcript-identity-cascade",
                str(normalized["conversation_id"]),
                str(normalized["recording_id"]),
                processing_version,
                str(cascade_ordinal),
            )
            con.execute(
                """
                INSERT INTO knowledge_transcript_identity_cascades (
                    id, conversation_id, recording_id, processing_version,
                    cascade_ordinal, triggering_generation_id, outcome,
                    content_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    cascade_id,
                    normalized["conversation_id"],
                    normalized["recording_id"],
                    processing_version,
                    cascade_ordinal,
                    normalized_generation_id,
                    outcome,
                    content_hash,
                    created_at,
                ),
            )
            con.commit()
        return IdentityCascadeReceipt(
            cascade_id=cascade_id,
            cascade_ordinal=cascade_ordinal,
            outcome=outcome,
            normalized_generation_id=normalized_generation_id,
            status="inserted",
        )

    def load_semantic_map(self, semantic_map_id: str) -> dict[str, Any]:
        with transcript_store.connect(self.root) as con:
            row = con.execute(
                """
                SELECT * FROM knowledge_transcript_semantic_maps WHERE id = ?
                """,
                (semantic_map_id,),
            ).fetchone()
        if row is None:
            raise ValueError("Transcript semantic map is unknown.")
        result = dict(row)
        result["sections"] = json.loads(str(row["map_json"]))
        result["transcript_only"] = bool(row["transcript_only"])
        return result

    def load_reindex_receipt(
        self,
        normalized_generation_id: str,
    ) -> dict[str, Any]:
        with transcript_store.connect(self.root) as con:
            row = con.execute(
                """
                SELECT *
                FROM knowledge_transcript_reindex_receipts
                WHERE normalized_generation_id = ?
                """,
                (normalized_generation_id,),
            ).fetchone()
        if row is None:
            raise ValueError("Transcript reindex receipt is unknown.")
        return dict(row)

    def load_normalized_generation(
        self,
        normalized_generation_id: str,
    ) -> dict[str, Any]:
        with transcript_store.connect(self.root) as con:
            row = con.execute(
                """
                SELECT *
                FROM knowledge_normalized_transcript_generations
                WHERE id = ?
                """,
                (normalized_generation_id,),
            ).fetchone()
        if row is None:
            raise ValueError("Normalized transcript generation is unknown.")
        result = dict(row)
        result["accepted_correction_ids"] = json.loads(
            str(row["accepted_correction_ids_json"])
        )
        result["raw_to_normalized_map"] = json.loads(
            str(row["raw_to_normalized_map_json"])
        )
        return result

    @classmethod
    def _validated_span(
        cls,
        value: object,
        text: str,
        *,
        kind: str,
    ) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise ValueError(f"Transcript semantic {kind} span is incomplete.")
        start = value.get("start")
        end = value.get("end")
        if (
            not isinstance(start, int)
            or not isinstance(end, int)
            or start < 0
            or end <= start
            or end > len(text)
        ):
            raise ValueError(f"Transcript semantic {kind} span is invalid.")
        span_hash = _text(value.get("text_sha256"))
        cls._require_sha256(span_hash, f"semantic_{kind}_text_sha256")
        actual_hash = hashlib.sha256(text[start:end].encode("utf-8")).hexdigest()
        if span_hash != actual_hash:
            raise ValueError(f"Transcript semantic {kind} span hash drifted.")
        return {"start": start, "end": end, "text_sha256": span_hash}

    @staticmethod
    def _select_corrections(
        rows: Sequence[Mapping[str, Any]],
    ) -> list[Mapping[str, Any]]:
        by_span: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
        for row in rows:
            by_span.setdefault(
                (int(row["span_start"]), int(row["span_end"])), []
            ).append(row)
        selected: list[Mapping[str, Any]] = []
        for span in sorted(by_span):
            candidates = by_span[span]
            best_rank = min(
                SCOPE_PRECEDENCE.index(str(row["scope_type"]))
                for row in candidates
            )
            best = [
                row
                for row in candidates
                if SCOPE_PRECEDENCE.index(str(row["scope_type"])) == best_rank
            ]
            replacements = {
                _normalized_term(row["replacement_text"]) for row in best
            }
            if len(replacements) > 1:
                raise ValueError(
                    "Equal-scope accepted transcript corrections conflict."
                )
            selected.append(sorted(best, key=lambda row: str(row["id"]))[0])
        for left, right in zip(selected, selected[1:]):
            if int(right["span_start"]) < int(left["span_end"]):
                raise ValueError("Accepted transcript correction spans overlap.")
        return selected

    @staticmethod
    def _apply_corrections(
        raw_text: str,
        corrections: Sequence[Mapping[str, Any]],
    ) -> tuple[str, list[dict[str, Any]]]:
        output: list[str] = []
        mapping: list[dict[str, Any]] = []
        raw_cursor = 0
        normalized_cursor = 0
        for correction in corrections:
            start = int(correction["span_start"])
            end = int(correction["span_end"])
            if start > raw_cursor:
                unchanged = raw_text[raw_cursor:start]
                output.append(unchanged)
                mapping.append(
                    {
                        "raw_start": raw_cursor,
                        "raw_end": start,
                        "normalized_start": normalized_cursor,
                        "normalized_end": normalized_cursor + len(unchanged),
                        "kind": "unchanged",
                    }
                )
                normalized_cursor += len(unchanged)
            replacement = str(correction["replacement_text"])
            output.append(replacement)
            mapping.append(
                {
                    "raw_start": start,
                    "raw_end": end,
                    "normalized_start": normalized_cursor,
                    "normalized_end": normalized_cursor + len(replacement),
                    "kind": "correction",
                    "correction_id": str(correction["id"]),
                }
            )
            normalized_cursor += len(replacement)
            raw_cursor = end
        if raw_cursor < len(raw_text):
            unchanged = raw_text[raw_cursor:]
            output.append(unchanged)
            mapping.append(
                {
                    "raw_start": raw_cursor,
                    "raw_end": len(raw_text),
                    "normalized_start": normalized_cursor,
                    "normalized_end": normalized_cursor + len(unchanged),
                    "kind": "unchanged",
                }
            )
        return "".join(output), mapping

    @staticmethod
    def _normalization_receipt(
        loaded: Mapping[str, Any],
        status: str,
    ) -> NormalizationReceipt:
        return NormalizationReceipt(
            normalized_generation_id=str(loaded["id"]),
            normalized_text=str(loaded["normalized_text"]),
            normalized_transcript_sha256=str(
                loaded["normalized_transcript_sha256"]
            ),
            accepted_correction_ids=tuple(loaded["accepted_correction_ids"]),
            correction_pass_count=int(loaded["correction_pass_count"]),
            index_version=str(loaded["index_version"]),
            status=status,
        )

    @staticmethod
    def _require_sha256(value: str, field_name: str) -> None:
        if not re.fullmatch(r"[a-f0-9]{64}", _text(value)):
            raise ValueError(f"{field_name} must be a lowercase SHA-256.")

    @staticmethod
    def _validate_terminology_entry(
        entry: TerminologyEntrySpec,
        version: str,
        created_at: str,
    ) -> None:
        if entry.scope_type not in SCOPE_PRECEDENCE or not _text(entry.scope_id):
            raise ValueError("Terminology entry scope is invalid.")
        if entry.scope_type == "global" and entry.scope_id != "global":
            raise ValueError("Global terminology scope ID must be global.")
        if entry.status not in {"draft", "reviewed", "rejected", "superseded"}:
            raise ValueError("Terminology entry status is invalid.")
        if not _text(entry.entry_id) or not _text(entry.canonical_term):
            raise ValueError("Terminology entry requires ID and canonical term.")
        validate_artifact(
            "terminology_entry",
            {
                "schema_version": ARTIFACT_SCHEMAS["terminology_entry"],
                "entry_id": entry.entry_id,
                "canonical_term": entry.canonical_term,
                "expansion": entry.expansion,
                "definition": entry.definition,
                "aliases": list(entry.aliases),
                "asr_confusions": list(entry.asr_confusions),
                "pronunciation_hints": list(entry.pronunciation_hints),
                "scope": {"type": entry.scope_type, "id": entry.scope_id},
                "source_observation_ids": list(entry.source_observation_ids),
                "status": entry.status,
                "version": version,
                "supersedes_entry_id": entry.supersedes_entry_id,
                "created_at": created_at,
            },
        )

    @staticmethod
    def _scope_applies(
        scope_type: str,
        scope_id: str,
        context: Mapping[str, str],
    ) -> bool:
        if scope_type == "global":
            return True
        return _text(context.get(scope_type)) == scope_id
