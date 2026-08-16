from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from uuid import NAMESPACE_URL, uuid5

import transcript_store
from identity_learning_contracts import (
    SUPERVISOR_STAGES,
    validate_adapter_exchange,
    validate_artifact,
)


PILLARS = (
    "calendar_association",
    "person_link",
    "contextual_speaker",
    "acoustic",
)
CALIBRATION_PILLARS = frozenset((*PILLARS, "combined"))
TRANSIENT_FAILURES = frozenset(
    {"transient_timeout", "transient_unavailable", "rate_limited"}
)


def _json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _hash(value: object) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _stable_id(prefix: str, *parts: str) -> str:
    return f"{prefix}-{uuid5(NAMESPACE_URL, chr(31).join(parts))}"


def _text(value: object) -> str:
    return str(value or "").strip()


def _strings(values: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{field_name} must be a list.")
    normalized = tuple(_text(value) for value in values)
    if any(not value for value in normalized) or len(normalized) != len(
        set(normalized)
    ):
        raise ValueError(f"{field_name} must contain unique non-empty values.")
    return normalized


@dataclass(frozen=True)
class SupervisorRunReceipt:
    run_id: str
    content_hash: str
    status: str


@dataclass(frozen=True)
class SupervisorStageReceipt:
    event_id: str
    run_id: str
    stage: str
    state: str
    status: str


@dataclass(frozen=True)
class AdapterExchangeReceipt:
    exchange_id: str
    run_id: str
    capability: str
    attempt: int
    status: str


@dataclass(frozen=True)
class EvidencePillarSpec:
    pillar: str
    score: float
    positive_factors: tuple[str, ...]
    negative_factors: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    independence_groups: tuple[str, ...]
    material_contradiction: bool = False


@dataclass(frozen=True)
class ScoreBatchReceipt:
    combined_assessment_id: str
    run_id: str
    candidate_id: str
    combined_score: float
    review_required: bool
    status: str


@dataclass(frozen=True)
class CalibrationOutcomeReceipt:
    outcome_id: str
    evaluation_version: str
    status: str


@dataclass(frozen=True)
class CalibratedLikelihoodReceipt:
    snapshot_id: str
    pillar: str
    score_band: str
    evaluation_version: str
    status: str
    sample_size: int
    likelihood: float | None
    interval_low: float | None
    interval_high: float | None


class IdentityEvidenceSupervisor:
    """Persist deterministic, zero-effect identity-evidence supervision."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = transcript_store.store_dir(root)
        with transcript_store.connect(self.root) as con:
            row = con.execute(
                "SELECT schema_version, dirty FROM knowledge_store_state "
                "WHERE singleton = 1"
            ).fetchone()
        if row is None or int(row["schema_version"]) < 7 or bool(row["dirty"]):
            raise RuntimeError("Identity evidence supervisor requires schema v7.")

    def start_run(self, payload: Mapping[str, Any]) -> SupervisorRunReceipt:
        artifact = validate_artifact("processing_run", payload)
        run_id = _text(artifact.get("run_id"))
        budgets = artifact.get("budgets")
        required_budgets = {
            "max_records",
            "max_characters",
            "max_calls",
            "max_latency_ms",
        }
        if (
            not run_id
            or artifact.get("operation_mode") not in {"contract_fixture", "shadow"}
            or artifact.get("state") != "running"
            or artifact.get("stage") != SUPERVISOR_STAGES[0]
            or not isinstance(budgets, Mapping)
            or not required_budgets.issubset(budgets)
            or any(
                not isinstance(budgets[field], int) or budgets[field] < 1
                for field in required_budgets
            )
        ):
            raise ValueError("Supervisor run start contract is invalid.")
        _strings(artifact.get("capabilities"), "run capabilities")
        source_scopes = artifact.get("source_scopes")
        if not isinstance(source_scopes, list) or not source_scopes:
            raise ValueError("Supervisor run requires exact source scopes.")
        for scope in source_scopes:
            self._normalize_source_scope(
                scope,
                allowed_capabilities=set(artifact["capabilities"]),
            )
        self._require_zero_effects(dict(artifact["effect_counts"]))
        core = dict(artifact)
        content_hash = _hash(core)
        with transcript_store.connect(self.root) as con:
            existing = con.execute(
                "SELECT content_hash FROM knowledge_identity_supervisor_runs "
                "WHERE id = ?",
                (run_id,),
            ).fetchone()
            if existing is not None:
                if str(existing["content_hash"]) != content_hash:
                    raise ValueError("Supervisor run already has different inputs.")
                return SupervisorRunReceipt(run_id, content_hash, "unchanged")
            con.execute("BEGIN IMMEDIATE")
            try:
                con.execute(
                    """
                    INSERT INTO knowledge_identity_supervisor_runs (
                        id, conversation_id, recording_id,
                        original_recording_filename, operation_mode,
                        artifact_json, content_hash, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        artifact["conversation_id"],
                        artifact["recording_id"],
                        artifact["original_recording_filename"],
                        artifact["operation_mode"],
                        _json(artifact),
                        content_hash,
                        artifact["created_at"],
                    ),
                )
                self._insert_stage_event(
                    con,
                    run_id=run_id,
                    stage=str(artifact["stage"]),
                    state=str(artifact["state"]),
                    output_ids=tuple(map(str, artifact["output_ids"])),
                    failures=tuple(artifact["failures"]),
                    effect_counts=dict(artifact["effect_counts"]),
                    idempotency_key=f"supervisor-run-start:{run_id}",
                    predecessor_event_id="",
                    created_at=str(artifact["created_at"]),
                )
                con.commit()
            except Exception:
                con.rollback()
                raise
        return SupervisorRunReceipt(run_id, content_hash, "inserted")

    def load_run(self, run_id: str) -> dict[str, Any]:
        with transcript_store.connect(self.root) as con:
            row = con.execute(
                "SELECT * FROM knowledge_identity_supervisor_runs WHERE id = ?",
                (_text(run_id),),
            ).fetchone()
            head = self._run_head(con, _text(run_id)) if row is not None else None
            retry_count = con.execute(
                "SELECT COUNT(*) AS count "
                "FROM knowledge_identity_supervisor_adapter_exchanges "
                "WHERE run_id = ? AND attempt = 1",
                (_text(run_id),),
            ).fetchone()
        if row is None or head is None:
            raise ValueError("Supervisor run is unknown.")
        artifact = json.loads(str(row["artifact_json"]))
        return {
            "artifact": artifact,
            "run_id": str(row["id"]),
            "stage": str(head["stage"]),
            "state": str(head["state"]),
            "event_id": str(head["id"]),
            "output_ids": json.loads(str(head["output_ids_json"])),
            "failures": json.loads(str(head["failures_json"])),
            "effect_counts": json.loads(str(head["effect_counts_json"])),
            "provider_retry_count": int(retry_count["count"]),
            "content_hash": str(row["content_hash"]),
        }

    def advance_stage(
        self,
        *,
        run_id: str,
        stage: str,
        state: str,
        output_ids: tuple[str, ...],
        failures: tuple[Mapping[str, Any], ...],
        effect_counts: Mapping[str, int],
        idempotency_key: str,
        created_at: str,
    ) -> SupervisorStageReceipt:
        run_id = _text(run_id)
        stage = _text(stage)
        state = _text(state)
        idempotency_key = _text(idempotency_key)
        created_at = _text(created_at)
        if not all((run_id, stage, state, idempotency_key, created_at)):
            raise ValueError("Supervisor stage event is incomplete.")
        normalized_outputs = _strings(output_ids, "stage output_ids")
        normalized_failures = tuple(dict(item) for item in failures)
        normalized_effects = dict(effect_counts)
        self._require_zero_effects(normalized_effects)
        with transcript_store.connect(self.root) as con:
            replay = con.execute(
                "SELECT * FROM knowledge_identity_supervisor_run_events "
                "WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
            if replay is not None:
                if (
                    str(replay["run_id"]) != run_id
                    or str(replay["stage"]) != stage
                    or str(replay["state"]) != state
                    or json.loads(str(replay["output_ids_json"]))
                    != list(normalized_outputs)
                    or json.loads(str(replay["failures_json"]))
                    != list(normalized_failures)
                    or json.loads(str(replay["effect_counts_json"]))
                    != normalized_effects
                ):
                    raise ValueError("Supervisor stage idempotency drifted.")
                return SupervisorStageReceipt(
                    str(replay["id"]), run_id, stage, state, "unchanged"
                )
            run = con.execute(
                "SELECT 1 FROM knowledge_identity_supervisor_runs WHERE id = ?",
                (run_id,),
            ).fetchone()
            head = self._run_head(con, run_id) if run is not None else None
            if head is None:
                raise ValueError("Supervisor run is unknown.")
            current_stage = str(head["stage"])
            if (
                current_stage not in SUPERVISOR_STAGES
                or stage not in SUPERVISOR_STAGES
                or SUPERVISOR_STAGES.index(stage)
                != SUPERVISOR_STAGES.index(current_stage) + 1
                or state != ("complete" if stage == "complete" else "running")
            ):
                raise ValueError("Supervisor stages must advance sequentially.")
            con.execute("BEGIN IMMEDIATE")
            try:
                receipt = self._insert_stage_event(
                    con,
                    run_id=run_id,
                    stage=stage,
                    state=state,
                    output_ids=normalized_outputs,
                    failures=normalized_failures,
                    effect_counts=normalized_effects,
                    idempotency_key=idempotency_key,
                    predecessor_event_id=str(head["id"]),
                    created_at=created_at,
                )
                con.commit()
            except Exception:
                con.rollback()
                raise
        return receipt

    def load_stage_history(self, run_id: str) -> tuple[dict[str, Any], ...]:
        with transcript_store.connect(self.root) as con:
            rows = con.execute(
                "SELECT * FROM knowledge_identity_supervisor_run_events "
                "WHERE run_id = ? ORDER BY rowid",
                (_text(run_id),),
            ).fetchall()
        return tuple(
            {
                "event_id": str(row["id"]),
                "stage": str(row["stage"]),
                "state": str(row["state"]),
                "output_ids": json.loads(str(row["output_ids_json"])),
                "failures": json.loads(str(row["failures_json"])),
                "effect_counts": json.loads(str(row["effect_counts_json"])),
                "predecessor_event_id": str(
                    row["predecessor_event_id"] or ""
                ),
                "created_at": str(row["created_at"]),
            }
            for row in rows
        )

    def record_adapter_exchange(
        self,
        request: Mapping[str, Any],
        result: Mapping[str, Any],
        *,
        attempt: int,
    ) -> AdapterExchangeReceipt:
        validated_request, validated_result = validate_adapter_exchange(
            request,
            result,
        )
        if attempt not in {0, 1}:
            raise ValueError("Provider retries are limited to one attempt.")
        run_id = str(validated_request["processing_run_id"])
        capability = str(validated_request["capability"])
        adapter_id = str(validated_request["adapter_id"])
        exchange_id = str(validated_request["request_id"])
        consumed = dict(validated_result["consumed_budget"])
        if (
            not isinstance(consumed.get("latency_ms"), int)
            or consumed["latency_ms"] < 0
            or consumed["latency_ms"]
            > int(validated_request["budgets"]["max_latency_ms"])
        ):
            raise ValueError("Adapter latency exceeded its request budget.")
        core = {
            "request": validated_request,
            "result": validated_result,
            "attempt": attempt,
        }
        content_hash = _hash(core)
        with transcript_store.connect(self.root) as con:
            existing = con.execute(
                "SELECT content_hash, status FROM "
                "knowledge_identity_supervisor_adapter_exchanges WHERE id = ?",
                (exchange_id,),
            ).fetchone()
            if existing is not None:
                if str(existing["content_hash"]) != content_hash:
                    raise ValueError("Adapter exchange idempotency drifted.")
                return AdapterExchangeReceipt(
                    exchange_id,
                    run_id,
                    capability,
                    attempt,
                    "unchanged",
                )
            run = con.execute(
                "SELECT artifact_json FROM knowledge_identity_supervisor_runs "
                "WHERE id = ?",
                (run_id,),
            ).fetchone()
            if run is None:
                raise ValueError("Adapter exchange references an unknown run.")
            artifact = json.loads(str(run["artifact_json"]))
            if validated_request["conversation_id"] != artifact["conversation_id"]:
                raise ValueError("Adapter exchange conversation is outside the run.")
            if capability not in artifact["capabilities"]:
                raise ValueError("Adapter capability is outside run capabilities.")
            requested_scope = self._normalize_source_scope(
                validated_request["source_scope"],
                allowed_capabilities=set(artifact["capabilities"]),
            )
            if not any(
                self._scope_allows(
                    configured,
                    requested_scope,
                    capability=capability,
                )
                for configured in artifact["source_scopes"]
            ):
                raise ValueError("Adapter scope is outside run source scopes.")
            prior_exchange_id = ""
            if attempt == 1:
                prior = con.execute(
                    """
                    SELECT * FROM knowledge_identity_supervisor_adapter_exchanges
                    WHERE run_id = ? AND adapter_id = ? AND capability = ?
                      AND attempt = 0
                    """,
                    (run_id, adapter_id, capability),
                ).fetchone()
                if prior is None:
                    raise ValueError("Adapter retry requires its initial exchange.")
                prior_result = json.loads(str(prior["result_json"]))
                failure = prior_result.get("failure")
                reason_code = (
                    _text(failure.get("reason_code"))
                    if isinstance(failure, Mapping)
                    else ""
                )
                if (
                    str(prior["status"]) not in {"partial", "unavailable"}
                    or reason_code not in TRANSIENT_FAILURES
                ):
                    raise ValueError("Only a transient adapter failure may retry.")
                prior_exchange_id = str(prior["id"])
            totals = con.execute(
                """
                SELECT COALESCE(SUM(consumed_records), 0) AS records,
                       COALESCE(SUM(consumed_characters), 0) AS characters,
                       COALESCE(SUM(consumed_calls), 0) AS calls,
                       COALESCE(SUM(consumed_latency_ms), 0) AS latency_ms
                FROM knowledge_identity_supervisor_adapter_exchanges
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
            limits = artifact["budgets"]
            prospective = {
                "records": int(totals["records"]) + int(consumed["records"]),
                "characters": int(totals["characters"])
                + int(consumed["characters"]),
                "calls": int(totals["calls"]) + int(consumed["calls"]),
                "latency_ms": int(totals["latency_ms"])
                + int(consumed["latency_ms"]),
            }
            limit_fields = {
                "records": "max_records",
                "characters": "max_characters",
                "calls": "max_calls",
                "latency_ms": "max_latency_ms",
            }
            if any(
                prospective[field] > int(limits[limit_field])
                for field, limit_field in limit_fields.items()
            ):
                raise ValueError("Adapter exchange exceeded the supervisor budget.")
            con.execute(
                """
                INSERT INTO knowledge_identity_supervisor_adapter_exchanges (
                    id, run_id, adapter_id, capability, attempt,
                    prior_exchange_id, request_json, result_json, status,
                    consumed_records, consumed_characters, consumed_calls,
                    consumed_latency_ms, content_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    exchange_id,
                    run_id,
                    adapter_id,
                    capability,
                    attempt,
                    prior_exchange_id or None,
                    _json(validated_request),
                    _json(validated_result),
                    validated_result["status"],
                    consumed["records"],
                    consumed["characters"],
                    consumed["calls"],
                    consumed["latency_ms"],
                    content_hash,
                    validated_result["retrieved_at"],
                ),
            )
            con.commit()
        return AdapterExchangeReceipt(
            exchange_id,
            run_id,
            capability,
            attempt,
            "inserted",
        )

    def load_adapter_exchanges(self, run_id: str) -> tuple[dict[str, Any], ...]:
        with transcript_store.connect(self.root) as con:
            rows = con.execute(
                "SELECT * FROM knowledge_identity_supervisor_adapter_exchanges "
                "WHERE run_id = ? ORDER BY rowid",
                (_text(run_id),),
            ).fetchall()
        return tuple(
            {
                "exchange_id": str(row["id"]),
                "attempt": int(row["attempt"]),
                "prior_exchange_id": str(row["prior_exchange_id"] or ""),
                "request": json.loads(str(row["request_json"])),
                "result": json.loads(str(row["result_json"])),
            }
            for row in rows
        )

    def record_conversation_candidate(
        self,
        run_id: str,
        payload: Mapping[str, Any],
    ) -> str:
        return self._record_hypothesis(
            run_id=run_id,
            artifact_kind="conversation_association_candidate",
            table="knowledge_conversation_association_candidates",
            id_field="candidate_id",
            payload=payload,
        )

    def record_participant_hypothesis(
        self,
        run_id: str,
        payload: Mapping[str, Any],
    ) -> str:
        return self._record_hypothesis(
            run_id=run_id,
            artifact_kind="participant_hypothesis",
            table="knowledge_participant_hypotheses",
            id_field="hypothesis_id",
            payload=payload,
        )

    def record_purpose_hypothesis(
        self,
        run_id: str,
        payload: Mapping[str, Any],
    ) -> str:
        required = {
            "schema_version",
            "hypothesis_id",
            "conversation_id",
            "label",
            "alternatives",
            "evidence_ids",
            "status",
            "created_at",
        }
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema_version")
            != "transcribe-audio.conversation-purpose-hypothesis.v1"
            or not required.issubset(payload)
            or payload.get("status") != "hypothesis"
            or not _text(payload.get("label"))
        ):
            raise ValueError("Conversation purpose hypothesis is invalid.")
        _strings(payload.get("alternatives"), "purpose alternatives")
        _strings(payload.get("evidence_ids"), "purpose evidence_ids")
        return self._record_hypothesis(
            run_id=run_id,
            artifact_kind="conversation_purpose_hypothesis",
            table="knowledge_conversation_purpose_hypotheses",
            id_field="hypothesis_id",
            payload=payload,
        )

    def load_hypotheses(self, run_id: str) -> dict[str, tuple[dict[str, Any], ...]]:
        with transcript_store.connect(self.root) as con:
            candidates = con.execute(
                "SELECT artifact_json FROM "
                "knowledge_conversation_association_candidates "
                "WHERE run_id = ? ORDER BY rowid",
                (_text(run_id),),
            ).fetchall()
            participants = con.execute(
                "SELECT artifact_json FROM knowledge_participant_hypotheses "
                "WHERE run_id = ? ORDER BY rowid",
                (_text(run_id),),
            ).fetchall()
            purposes = con.execute(
                "SELECT artifact_json FROM "
                "knowledge_conversation_purpose_hypotheses "
                "WHERE run_id = ? ORDER BY rowid",
                (_text(run_id),),
            ).fetchall()
        return {
            "calendar_candidates": tuple(
                json.loads(str(row["artifact_json"])) for row in candidates
            ),
            "participants": tuple(
                json.loads(str(row["artifact_json"])) for row in participants
            ),
            "purposes": tuple(
                json.loads(str(row["artifact_json"])) for row in purposes
            ),
        }

    def score_candidate(
        self,
        *,
        run_id: str,
        candidate_id: str,
        pillars: tuple[EvidencePillarSpec, ...],
        rubric_version: str,
        model_version: str,
        created_at: str,
        predecessor_assessment_id: str = "",
    ) -> ScoreBatchReceipt:
        run_id = _text(run_id)
        candidate_id = _text(candidate_id)
        rubric_version = _text(rubric_version)
        model_version = _text(model_version)
        created_at = _text(created_at)
        if not all(
            (run_id, candidate_id, rubric_version, model_version, created_at)
        ):
            raise ValueError("Evidence assessment identity is incomplete.")
        prepared = self._prepare_pillars(pillars)
        contradiction = any(item["material_contradiction"] for item in prepared)
        all_groups = [
            group
            for item in prepared
            for group in item["independence_groups"]
        ]
        duplicate_groups = len(all_groups) != len(set(all_groups))
        raw_combined = round(
            sum(float(item["score"]) for item in prepared) / len(prepared),
            2,
        )
        review_required = contradiction or duplicate_groups
        combined_score = min(raw_combined, 49.0) if review_required else raw_combined
        reason_codes = []
        if contradiction:
            reason_codes.append("material_contradiction_cap")
        if duplicate_groups:
            reason_codes.append("duplicate_evidence_group_cap")
        predecessor_assessment_id = _text(predecessor_assessment_id)
        core = {
            "run_id": run_id,
            "candidate_id": candidate_id,
            "predecessor_assessment_id": predecessor_assessment_id,
            "rubric_version": rubric_version,
            "model_version": model_version,
            "pillar_assessments": prepared,
            "combined_score": combined_score,
            "review_required": review_required,
            "reason_codes": reason_codes,
        }
        content_hash = _hash(core)
        batch_id = _stable_id(
            "evidence-assessment",
            run_id,
            candidate_id,
            rubric_version,
        )
        with transcript_store.connect(self.root) as con:
            existing = con.execute(
                "SELECT content_hash FROM knowledge_evidence_assessment_batches "
                "WHERE id = ?",
                (batch_id,),
            ).fetchone()
            if existing is not None:
                if str(existing["content_hash"]) != content_hash:
                    raise ValueError("Evidence assessment idempotency drifted.")
                return ScoreBatchReceipt(
                    batch_id,
                    run_id,
                    candidate_id,
                    combined_score,
                    review_required,
                    "unchanged",
                )
            run = con.execute(
                "SELECT 1 FROM knowledge_identity_supervisor_runs WHERE id = ?",
                (run_id,),
            ).fetchone()
            history = con.execute(
                "SELECT id FROM knowledge_evidence_assessment_batches "
                "WHERE run_id = ? AND candidate_id = ? ORDER BY rowid",
                (run_id, candidate_id),
            ).fetchall()
            if run is None:
                raise ValueError("Evidence assessment references an unknown run.")
            expected_predecessor = str(history[-1]["id"]) if history else ""
            if predecessor_assessment_id != expected_predecessor:
                raise ValueError(
                    "Evidence assessment must cite its exact predecessor."
                )
            con.execute("BEGIN IMMEDIATE")
            try:
                con.execute(
                    """
                    INSERT INTO knowledge_evidence_assessment_batches (
                        id, run_id, candidate_id, predecessor_assessment_id,
                        rubric_version, model_version, combined_score,
                        review_required, reason_codes_json, content_hash,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        batch_id,
                        run_id,
                        candidate_id,
                        predecessor_assessment_id or None,
                        rubric_version,
                        model_version,
                        combined_score,
                        int(review_required),
                        _json(reason_codes),
                        content_hash,
                        created_at,
                    ),
                )
                for item in prepared:
                    pillar_hash = _hash({"batch_id": batch_id, **item})
                    con.execute(
                        """
                        INSERT INTO knowledge_evidence_pillar_assessments (
                            id, batch_id, pillar, score,
                            positive_factors_json, negative_factors_json,
                            evidence_ids_json, independence_groups_json,
                            material_contradiction, content_hash, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            _stable_id(
                                "evidence-pillar", batch_id, item["pillar"]
                            ),
                            batch_id,
                            item["pillar"],
                            item["score"],
                            _json(item["positive_factors"]),
                            _json(item["negative_factors"]),
                            _json(item["evidence_ids"]),
                            _json(item["independence_groups"]),
                            int(item["material_contradiction"]),
                            pillar_hash,
                            created_at,
                        ),
                    )
                con.commit()
            except Exception:
                con.rollback()
                raise
        return ScoreBatchReceipt(
            batch_id,
            run_id,
            candidate_id,
            combined_score,
            review_required,
            "inserted",
        )

    def load_assessment_history(
        self,
        run_id: str,
        candidate_id: str,
    ) -> tuple[dict[str, Any], ...]:
        with transcript_store.connect(self.root) as con:
            batches = con.execute(
                "SELECT * FROM knowledge_evidence_assessment_batches "
                "WHERE run_id = ? AND candidate_id = ? ORDER BY rowid",
                (_text(run_id), _text(candidate_id)),
            ).fetchall()
            result: list[dict[str, Any]] = []
            for batch in batches:
                pillars = con.execute(
                    "SELECT * FROM knowledge_evidence_pillar_assessments "
                    "WHERE batch_id = ? ORDER BY rowid",
                    (batch["id"],),
                ).fetchall()
                result.append(
                    {
                        "assessment_id": str(batch["id"]),
                        "predecessor_assessment_id": str(
                            batch["predecessor_assessment_id"] or ""
                        ),
                        "rubric_version": str(batch["rubric_version"]),
                        "model_version": str(batch["model_version"]),
                        "combined_score": float(batch["combined_score"]),
                        "review_required": bool(batch["review_required"]),
                        "reason_codes": json.loads(
                            str(batch["reason_codes_json"])
                        ),
                        "pillar_assessments": [
                            {
                                "pillar": str(item["pillar"]),
                                "score": float(item["score"]),
                                "positive_factors": json.loads(
                                    str(item["positive_factors_json"])
                                ),
                                "negative_factors": json.loads(
                                    str(item["negative_factors_json"])
                                ),
                                "evidence_ids": json.loads(
                                    str(item["evidence_ids_json"])
                                ),
                                "independence_groups": json.loads(
                                    str(item["independence_groups_json"])
                                ),
                                "material_contradiction": bool(
                                    item["material_contradiction"]
                                ),
                            }
                            for item in pillars
                        ],
                    }
                )
        return tuple(result)

    def record_calibration_outcome(
        self,
        *,
        pillar: str,
        score_band: str,
        correct: bool,
        source_disjoint_id: str,
        evaluation_version: str,
        review_decision_id: str,
        created_at: str,
    ) -> CalibrationOutcomeReceipt:
        pillar = _text(pillar)
        score_band = _text(score_band)
        source_disjoint_id = _text(source_disjoint_id)
        evaluation_version = _text(evaluation_version)
        review_decision_id = _text(review_decision_id)
        created_at = _text(created_at)
        if (
            pillar not in CALIBRATION_PILLARS
            or not all(
                (
                    score_band,
                    source_disjoint_id,
                    evaluation_version,
                    review_decision_id,
                    created_at,
                )
            )
            or not isinstance(correct, bool)
        ):
            raise ValueError("Calibration outcome is invalid.")
        core = {
            "pillar": pillar,
            "score_band": score_band,
            "correct": correct,
            "source_disjoint_id": source_disjoint_id,
            "evaluation_version": evaluation_version,
            "review_decision_id": review_decision_id,
        }
        content_hash = _hash(core)
        outcome_id = _stable_id(
            "calibration-outcome",
            evaluation_version,
            pillar,
            score_band,
            source_disjoint_id,
        )
        with transcript_store.connect(self.root) as con:
            existing = con.execute(
                "SELECT id, content_hash FROM "
                "knowledge_evidence_calibration_outcomes "
                "WHERE evaluation_version = ? AND pillar = ? "
                "AND score_band = ? AND source_disjoint_id = ?",
                (evaluation_version, pillar, score_band, source_disjoint_id),
            ).fetchone()
            if existing is not None:
                if str(existing["content_hash"]) != content_hash:
                    raise ValueError(
                        "Calibration source-disjoint outcome already differs."
                    )
                return CalibrationOutcomeReceipt(
                    str(existing["id"]), evaluation_version, "unchanged"
                )
            con.execute(
                """
                INSERT INTO knowledge_evidence_calibration_outcomes (
                    id, pillar, score_band, correct, source_disjoint_id,
                    evaluation_version, review_decision_id, content_hash,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    outcome_id,
                    pillar,
                    score_band,
                    int(correct),
                    source_disjoint_id,
                    evaluation_version,
                    review_decision_id,
                    content_hash,
                    created_at,
                ),
            )
            con.commit()
        return CalibrationOutcomeReceipt(outcome_id, evaluation_version, "inserted")

    def calibrated_likelihood(
        self,
        *,
        pillar: str,
        score_band: str,
        evaluation_version: str,
        created_at: str,
    ) -> CalibratedLikelihoodReceipt:
        pillar = _text(pillar)
        score_band = _text(score_band)
        evaluation_version = _text(evaluation_version)
        created_at = _text(created_at)
        if (
            pillar not in CALIBRATION_PILLARS
            or not all((score_band, evaluation_version, created_at))
        ):
            raise ValueError("Calibration likelihood request is invalid.")
        with transcript_store.connect(self.root) as con:
            rows = con.execute(
                "SELECT id, correct, content_hash FROM "
                "knowledge_evidence_calibration_outcomes "
                "WHERE evaluation_version = ? AND pillar = ? "
                "AND score_band = ? ORDER BY rowid",
                (evaluation_version, pillar, score_band),
            ).fetchall()
        sample_size = len(rows)
        successes = sum(int(row["correct"]) for row in rows)
        watermark = _hash(
            [
                {"id": str(row["id"]), "content_hash": str(row["content_hash"])}
                for row in rows
            ]
        )
        status = "available" if sample_size >= 30 else "insufficient_data"
        likelihood: float | None = None
        interval_low: float | None = None
        interval_high: float | None = None
        if status == "available":
            likelihood = successes / sample_size
            interval_low, interval_high = self._wilson_interval(
                successes,
                sample_size,
            )
        core = {
            "pillar": pillar,
            "score_band": score_band,
            "evaluation_version": evaluation_version,
            "input_watermark": watermark,
            "status": status,
            "sample_size": sample_size,
            "likelihood": likelihood,
            "interval_low": interval_low,
            "interval_high": interval_high,
        }
        content_hash = _hash(core)
        snapshot_id = _stable_id(
            "calibration-snapshot",
            evaluation_version,
            pillar,
            score_band,
            watermark,
        )
        with transcript_store.connect(self.root) as con:
            existing = con.execute(
                "SELECT content_hash FROM "
                "knowledge_evidence_calibration_snapshots WHERE id = ?",
                (snapshot_id,),
            ).fetchone()
            if existing is None:
                con.execute(
                    """
                    INSERT INTO knowledge_evidence_calibration_snapshots (
                        id, pillar, score_band, evaluation_version,
                        input_watermark, status, sample_size, likelihood,
                        interval_low, interval_high, content_hash, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        snapshot_id,
                        pillar,
                        score_band,
                        evaluation_version,
                        watermark,
                        status,
                        sample_size,
                        likelihood,
                        interval_low,
                        interval_high,
                        content_hash,
                        created_at,
                    ),
                )
                con.commit()
            elif str(existing["content_hash"]) != content_hash:
                raise ValueError("Calibration snapshot idempotency drifted.")
        return CalibratedLikelihoodReceipt(
            snapshot_id,
            pillar,
            score_band,
            evaluation_version,
            status,
            sample_size,
            likelihood,
            interval_low,
            interval_high,
        )

    def _record_hypothesis(
        self,
        *,
        run_id: str,
        artifact_kind: str,
        table: str,
        id_field: str,
        payload: Mapping[str, Any],
    ) -> str:
        artifact = (
            dict(payload)
            if artifact_kind == "conversation_purpose_hypothesis"
            else validate_artifact(artifact_kind, payload)
        )
        run_id = _text(run_id)
        artifact_id = _text(artifact.get(id_field))
        if not run_id or not artifact_id:
            raise ValueError("Supervisor hypothesis identity is incomplete.")
        content_hash = _hash(artifact)
        with transcript_store.connect(self.root) as con:
            run = con.execute(
                "SELECT conversation_id, recording_id "
                "FROM knowledge_identity_supervisor_runs WHERE id = ?",
                (run_id,),
            ).fetchone()
            if run is None:
                raise ValueError("Supervisor hypothesis references an unknown run.")
            if (
                artifact.get("conversation_id") != run["conversation_id"]
                or artifact_kind == "conversation_association_candidate"
                and artifact.get("recording_id") != run["recording_id"]
            ):
                raise ValueError("Supervisor hypothesis is outside the run scope.")
            existing = con.execute(
                f"SELECT content_hash FROM {table} WHERE id = ?",
                (artifact_id,),
            ).fetchone()
            if existing is not None:
                if str(existing["content_hash"]) != content_hash:
                    raise ValueError("Supervisor hypothesis idempotency drifted.")
                return "unchanged"
            con.execute(
                f"""
                INSERT INTO {table} (
                    id, run_id, artifact_json, content_hash, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    artifact_id,
                    run_id,
                    _json(artifact),
                    content_hash,
                    artifact["created_at"],
                ),
            )
            con.commit()
        return "inserted"

    @staticmethod
    def _prepare_pillars(
        pillars: tuple[EvidencePillarSpec, ...],
    ) -> list[dict[str, Any]]:
        if {item.pillar for item in pillars} != set(PILLARS) or len(pillars) != len(
            PILLARS
        ):
            raise ValueError("Evidence assessment requires every visible pillar.")
        prepared: list[dict[str, Any]] = []
        for pillar in PILLARS:
            item = next(value for value in pillars if value.pillar == pillar)
            if (
                isinstance(item.score, bool)
                or not isinstance(item.score, (int, float))
                or not 0 <= float(item.score) <= 100
            ):
                raise ValueError("Evidence pillar score must be between 0 and 100.")
            positive = _strings(item.positive_factors, "positive_factors")
            negative = _strings(item.negative_factors, "negative_factors")
            evidence_ids = _strings(item.evidence_ids, "evidence_ids")
            groups = _strings(
                item.independence_groups,
                "independence_groups",
            )
            if not positive and not negative:
                raise ValueError("Evidence pillar requires scored factors.")
            prepared.append(
                {
                    "pillar": pillar,
                    "score": float(item.score),
                    "positive_factors": list(positive),
                    "negative_factors": list(negative),
                    "evidence_ids": list(evidence_ids),
                    "independence_groups": list(groups),
                    "material_contradiction": bool(
                        item.material_contradiction
                    ),
                }
            )
        return prepared

    @staticmethod
    def _require_zero_effects(effect_counts: Mapping[str, int]) -> None:
        if not effect_counts or any(
            not isinstance(value, int) or value != 0
            for value in effect_counts.values()
        ):
            raise ValueError("A4 supervisor events require zero effects.")

    @staticmethod
    def _normalize_source_scope(
        scope: object,
        *,
        allowed_capabilities: set[str],
    ) -> dict[str, Any]:
        required = {
            "provider_kind",
            "profile_id",
            "account_id",
            "tenant_id",
            "capabilities",
        }
        if not isinstance(scope, Mapping) or not required.issubset(scope):
            raise ValueError("Supervisor source scope is incomplete.")
        selectors = {
            field: _text(scope.get(field))
            for field in required - {"capabilities"}
        }
        capabilities = _strings(
            scope.get("capabilities"),
            "source scope capabilities",
        )
        if (
            not all(selectors.values())
            or not set(capabilities).issubset(allowed_capabilities)
        ):
            raise ValueError("Supervisor source scope is outside capabilities.")
        return {**selectors, "capabilities": list(capabilities)}

    @classmethod
    def _scope_allows(
        cls,
        configured: object,
        requested: Mapping[str, Any],
        *,
        capability: str,
    ) -> bool:
        try:
            normalized = cls._normalize_source_scope(
                configured,
                allowed_capabilities=set(configured.get("capabilities", ())),
            )
        except (AttributeError, ValueError):
            return False
        selector_fields = (
            "provider_kind",
            "profile_id",
            "account_id",
            "tenant_id",
        )
        return (
            all(normalized[field] == requested[field] for field in selector_fields)
            and capability in normalized["capabilities"]
            and capability in requested["capabilities"]
        )

    @staticmethod
    def _run_head(con: Any, run_id: str) -> Any:
        rows = con.execute(
            """
            SELECT event.* FROM knowledge_identity_supervisor_run_events event
            WHERE event.run_id = ? AND NOT EXISTS (
                SELECT 1 FROM knowledge_identity_supervisor_run_events successor
                WHERE successor.predecessor_event_id = event.id
            )
            """,
            (run_id,),
        ).fetchall()
        if len(rows) > 1:
            raise RuntimeError("Supervisor run history has multiple heads.")
        return rows[0] if rows else None

    def _insert_stage_event(
        self,
        con: Any,
        *,
        run_id: str,
        stage: str,
        state: str,
        output_ids: tuple[str, ...],
        failures: tuple[Mapping[str, Any], ...],
        effect_counts: Mapping[str, int],
        idempotency_key: str,
        predecessor_event_id: str,
        created_at: str,
    ) -> SupervisorStageReceipt:
        self._require_zero_effects(effect_counts)
        core = {
            "run_id": run_id,
            "stage": stage,
            "state": state,
            "output_ids": list(output_ids),
            "failures": list(failures),
            "effect_counts": dict(effect_counts),
            "idempotency_key": idempotency_key,
            "predecessor_event_id": predecessor_event_id,
        }
        content_hash = _hash(core)
        event_id = _stable_id("supervisor-run-event", idempotency_key)
        con.execute(
            """
            INSERT INTO knowledge_identity_supervisor_run_events (
                id, run_id, stage, state, output_ids_json, failures_json,
                effect_counts_json, idempotency_key, predecessor_event_id,
                content_hash, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                run_id,
                stage,
                state,
                _json(list(output_ids)),
                _json(list(failures)),
                _json(dict(effect_counts)),
                idempotency_key,
                predecessor_event_id or None,
                content_hash,
                created_at,
            ),
        )
        return SupervisorStageReceipt(event_id, run_id, stage, state, "inserted")

    @staticmethod
    def _wilson_interval(successes: int, sample_size: int) -> tuple[float, float]:
        z = 1.96
        proportion = successes / sample_size
        denominator = 1 + z * z / sample_size
        center = (
            proportion + z * z / (2 * sample_size)
        ) / denominator
        spread = (
            z
            * math.sqrt(
                proportion * (1 - proportion) / sample_size
                + z * z / (4 * sample_size * sample_size)
            )
            / denominator
        )
        return max(0.0, center - spread), min(1.0, center + spread)
