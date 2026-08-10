"""Plan 0064 P2 governed contextual-evidence reuse.

Each frozen P0 recording is processed by the existing two-phase clue discovery
and identity-evaluation workflow.  The host owns sequencing, schema validation,
bounded reference repair, durable run references, and immutable per-recording
checkpoints.  Model output cannot directly assign a speaker or mutate contact
providers.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import json
import os
from pathlib import Path
from typing import Any
from uuid import UUID

import requests

from acoustic_audio_derivatives import (
    canonical_artifact_hash,
    ensure_private_tree,
    read_private_object,
    require_private_file,
    sha256_file,
    write_immutable_private_json,
)
import app_intelligence_ledger
from speaker_evaluation_baseline import CasePredictionFailure, LocalSpeakerCaseRunner
from speaker_identity_plan0064_p0 import (
    DEFAULT_RUNTIME_ROOT,
    Plan0064P0Error,
    _paths as p0_paths,
    _validate_manifest as validate_p0_manifest,
    build_p0_manifest,
    replay_p0,
)


P2_SCHEMA = "transcribe-audio.plan0064-p2-context-evidence.v1"
CASE_SCHEMA = "transcribe-audio.plan0064-p2-context-case.v1"
RECEIPT_SCHEMA = "transcribe-audio.plan0064-p2-context-receipt.v1"
ACTION_COUNTS = {
    "speaker_assignments": 0,
    "new_enrollments": 0,
    "profile_mutations": 0,
    "knowledge_writes": 0,
    "provider_writes": 0,
    "external_provider_writes": 0,
}
HYDRATION_BRIDGE_SCHEMA = "transcribe-audio.plan0064-p0-identity-hydration-bridge.v1"
HYDRATION_FIELDS = frozenset(
    {"artifact_sha256", "conversation_id", "recording_id", "transcript_artifact"}
)


class Plan0064P2Error(ValueError):
    """Raised when P2 authority, model evidence, or replay drifts."""


class OpenAICompatibleCaseRunner(LocalSpeakerCaseRunner):
    """Configured direct-HTTP fallback when the primary app-server is unavailable."""

    def __init__(
        self,
        *,
        base_url: str = "",
        api_key: str = "",
        model: str = "gpt-4o-mini",
        timeout_seconds: float = 180.0,
        state_root: Path = Path("~/.local/state/transcribe-audio"),
    ) -> None:
        super().__init__(timeout_seconds=timeout_seconds)
        self.openai_base_url = (
            base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        ).rstrip("/")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or ""
        self.model = model
        self.state_root = state_root.expanduser().absolute()
        if not self.api_key:
            raise Plan0064P2Error("The configured OpenAI-compatible fallback lacks an API key.")

    def _direct_readout(self, prepared: Mapping[str, Any]) -> dict[str, Any]:
        run_id = str(prepared.get("run_id") or "")
        packet = prepared.get("prompt_packet")
        packet_prompt_path = (
            str(packet.get("prompt_path") or "")
            if isinstance(packet, Mapping)
            else ""
        )
        prompt_path = str(prepared.get("prompt_path") or "")
        packet_path = str(prepared.get("packet_path") or "")
        if (
            not run_id
            or not prompt_path
            or prompt_path != packet_prompt_path
            or not packet_path
        ):
            raise Plan0064P2Error("The fallback run lacks a prepared host prompt.")
        selected_paths = [
            Path(packet_path).expanduser().absolute(),
            Path(prompt_path).expanduser().absolute(),
        ]
        selected_root = self.state_root.resolve(strict=True)
        for path in selected_paths:
            resolved = path.resolve(strict=True)
            try:
                resolved.relative_to(selected_root)
            except ValueError as exc:
                raise Plan0064P2Error(
                    "A fallback prompt artifact escaped the private state root."
                ) from exc
            if path.is_symlink() or not path.is_file():
                raise Plan0064P2Error(
                    "A fallback prompt artifact is not a regular file."
                )
            relative_parent = resolved.parent.relative_to(selected_root)
            current_parent = selected_root
            for part in relative_parent.parts:
                current_parent = current_parent / part
                if current_parent.is_symlink() or not current_parent.is_dir():
                    raise Plan0064P2Error(
                        "A fallback prompt artifact has an unsafe parent directory."
                    )
                current_parent.chmod(0o700)
            path.chmod(0o600)
            require_private_file(path, self.state_root)
        selected_prompt_path = selected_paths[1]
        prompt = selected_prompt_path.read_text(encoding="utf-8")
        if not prompt.strip():
            raise Plan0064P2Error("The fallback host prompt is empty.")
        app_intelligence_ledger.append_event(
            state_root=self.state_root,
            run_id=run_id,
            event_type="model_turn_fallback_started",
            payload={
                "provider": "openai-compatible",
                "model": self.model,
                "host_owns_control_flow": True,
                "will_execute_downstream_action": False,
            },
        )
        try:
            response = requests.post(
                f"{self.openai_base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "temperature": 0,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {
                            "role": "system",
                            "content": "Return only the requested JSON object. Do not execute actions.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                },
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise Plan0064P2Error("The OpenAI-compatible fallback request failed.") from exc
        if response.status_code >= 400:
            raise Plan0064P2Error(
                f"The OpenAI-compatible fallback returned HTTP {response.status_code}."
            )
        try:
            payload = response.json()
            content = str(payload["choices"][0]["message"]["content"])
            readout = app_intelligence_ledger.extract_json_object(content)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise Plan0064P2Error("The fallback response was not a JSON object.") from exc
        app_intelligence_ledger.append_event(
            state_root=self.state_root,
            run_id=run_id,
            event_type="model_turn_fallback_completed",
            payload={
                "provider": "openai-compatible",
                "model": self.model,
                "output_sha256": _hash(readout),
                "will_execute_downstream_action": False,
            },
        )
        return readout

    def __call__(self, document_id: str) -> dict[str, Any]:
        discovery = self._post(
            f"/api/conversations/{document_id}/speaker-preprocessing/prepare-discovery",
            {},
        )
        run_references = {"clue_discovery_run_id": discovery["run_id"]}
        discovery_readout = self._direct_readout(discovery)
        try:
            evaluation = self._post(
                f"/api/conversations/{document_id}/speaker-preprocessing/prepare-evaluation",
                {
                    "clue_discovery_run_id": discovery["run_id"],
                    "discovery_readout": discovery_readout,
                },
            )
        except ValueError as exc:
            raise CasePredictionFailure(
                "fallback_clue_discovery_validation",
                str(exc),
                run_references=run_references,
            ) from exc
        run_references["identity_evaluation_run_id"] = evaluation["run_id"]
        evaluation_readout = self._direct_readout(evaluation)
        try:
            persisted = self._post(
                f"/api/conversations/{document_id}/speaker-preprocessing/capture-evaluation",
                {**run_references, "readout": evaluation_readout},
            )
        except ValueError as exc:
            raise CasePredictionFailure(
                "fallback_identity_evaluation_validation",
                str(exc),
                run_references=run_references,
            ) from exc
        record = persisted.get("record") if isinstance(persisted.get("record"), dict) else {}
        current_id = str(record.get("current_evaluation_id") or "")
        prediction = next(
            (
                item
                for item in record.get("evaluations") or []
                if isinstance(item, dict) and str(item.get("evaluation_id") or "") == current_id
            ),
            None,
        )
        if prediction is None:
            raise Plan0064P2Error("Fallback capture did not expose its current evaluation.")
        return {
            "prediction": prediction,
            "run_references": run_references,
            "execution_provider": "openai-compatible",
        }


def _hash(value: Any) -> str:
    return canonical_artifact_hash(value)


def _content_addressed(value: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(value)
    body.pop("content_sha256", None)
    return {**body, "content_sha256": _hash(body)}


def _read_object(path: Path) -> dict[str, Any]:
    value = read_private_object(path)
    if not isinstance(value, dict):
        raise Plan0064P2Error("A governed artifact is not a JSON object.")
    return value


def _uuid(value: Any) -> str:
    try:
        return str(UUID(str(value or "")))
    except (ValueError, AttributeError, TypeError) as exc:
        raise Plan0064P2Error("Hydrated transcript identity is not a UUID.") from exc


def _phase_safe_p0(
    p0_content_sha256: str, *, runtime_root: Path
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Replay P0 or prove the workflow's exact ID-only transcript hydration."""
    try:
        return (
            replay_p0(content_sha256=p0_content_sha256, runtime_root=runtime_root),
            None,
        )
    except Plan0064P0Error as exc:
        if exc.reason_code != "p0_live_state_drift":
            raise
    paths = p0_paths(runtime_root, p0_content_sha256)
    require_private_file(paths["manifest"], paths["root"])
    require_private_file(paths["receipt"], paths["root"])
    frozen = _read_object(paths["manifest"])
    receipt = _read_object(paths["receipt"])
    if (
        validate_p0_manifest(frozen) != p0_content_sha256
        or receipt.get("manifest_content_sha256") != p0_content_sha256
        or receipt.get("manifest_file_sha256") != sha256_file(paths["manifest"])
        or any(receipt.get("action_counts", {}).values())
    ):
        raise Plan0064P2Error("The frozen P0 authority drifted before hydration review.")
    current = build_p0_manifest(repository=frozen["repository_authority"])
    for key in sorted(set(frozen) | set(current)):
        if key not in {"content_sha256", "evaluation_cohort"} and frozen.get(key) != current.get(key):
            raise Plan0064P2Error(f"P0 drift outside transcript identity hydration: {key}.")
    old_cohort = frozen["evaluation_cohort"]
    new_cohort = current["evaluation_cohort"]
    for key in sorted(set(old_cohort) | set(new_cohort)):
        if key not in {"cohort_sha256", "considered"} and old_cohort.get(key) != new_cohort.get(key):
            raise Plan0064P2Error(f"P0 cohort drift outside transcript hydration: {key}.")
    old_rows, new_rows = old_cohort["considered"], new_cohort["considered"]
    if len(old_rows) != len(new_rows):
        raise Plan0064P2Error("P0 cohort length changed during transcript hydration.")
    changed = []
    normalized_payload_hashes = []
    for old, new in zip(old_rows, new_rows):
        if old.get("document_id") != new.get("document_id"):
            raise Plan0064P2Error("P0 chronological document order changed.")
        fields = {
            key for key in set(old) | set(new) if old.get(key) != new.get(key)
        }
        if not fields:
            continue
        if not fields <= HYDRATION_FIELDS or old.get("disposition") != "selected_evaluation_candidate":
            raise Plan0064P2Error("A P0 row changed outside sanctioned identity hydration.")
        old_artifact = old.get("transcript_artifact") or {}
        new_artifact = new.get("transcript_artifact") or {}
        if old_artifact.get("path") != new_artifact.get("path"):
            raise Plan0064P2Error("Transcript hydration changed the governed artifact path.")
        transcript_path = Path(str(new_artifact.get("path") or ""))
        payload = json.loads(transcript_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise Plan0064P2Error("A hydrated transcript is not a JSON object.")
        conversation_id = _uuid(payload.get("conversation_id"))
        recording_id = _uuid(payload.get("recording_id"))
        if (
            conversation_id != new.get("conversation_id")
            or recording_id != new.get("recording_id")
            or int(payload.get("schema_version") or 0) < 2
            or sha256_file(transcript_path) != new.get("artifact_sha256")
        ):
            raise Plan0064P2Error("Hydrated transcript identity is not synchronized.")
        normalized = dict(payload)
        normalized.pop("conversation_id", None)
        normalized.pop("recording_id", None)
        normalized.pop("schema_version", None)
        normalized_payload_hashes.append(_hash(normalized))
        changed.append(
            {
                "document_id_sha256": _hash(str(old["document_id"])),
                "old_artifact_sha256": old["artifact_sha256"],
                "new_artifact_sha256": new["artifact_sha256"],
                "conversation_id_sha256": _hash(conversation_id),
                "recording_id_sha256": _hash(recording_id),
                "changed_fields": sorted(fields),
            }
        )
    if not changed:
        raise Plan0064P2Error("P0 replay failed without an identity-hydration delta.")
    bridge = _content_addressed(
        {
            "schema_version": HYDRATION_BRIDGE_SCHEMA,
            "status": "validated_transcript_identity_hydration",
            "p0_content_sha256": p0_content_sha256,
            "old_cohort_sha256": old_cohort["cohort_sha256"],
            "current_cohort_sha256": new_cohort["cohort_sha256"],
            "changed_recording_count": len(changed),
            "changed_rows": changed,
            "normalized_payload_set_sha256": _hash(sorted(normalized_payload_hashes)),
            "preserved_recording_count": old_cohort["selected_count"],
            "preserved_speaker_slot_count": sum(
                len(row["speaker_labels"])
                for row in old_rows
                if row["disposition"] == "selected_evaluation_candidate"
            ),
            "allowed_fields": sorted(HYDRATION_FIELDS),
            "action_counts": dict(ACTION_COUNTS),
        }
    )
    replay = {
        "status": "p0_frozen_with_validated_identity_hydration",
        "content_sha256": p0_content_sha256,
        "receipt_content_sha256": receipt["content_sha256"],
        "private_manifest_path": str(paths["manifest"]),
        "private_receipt_path": str(paths["receipt"]),
        "idempotent_replay": True,
    }
    return replay, bridge


def build_p2_preview(
    p0_content_sha256: str,
    *,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
) -> dict[str, Any]:
    """Bind P2 to P0, current canonical bindings, and the full slot denominator."""
    p0, _bridge = _phase_safe_p0(p0_content_sha256, runtime_root=runtime_root)
    manifest = _read_object(Path(p0["private_manifest_path"]))
    selected = [
        item
        for item in manifest["evaluation_cohort"]["considered"]
        if item["disposition"] == "selected_evaluation_candidate"
    ]
    people = sorted(
        item["person_id"]
        for item in manifest["canonical_bindings"]["subject_bindings"]
        if item["identity_candidate_eligible"]
    )
    if len(selected) != 12 or len(people) != 6:
        raise Plan0064P2Error("The frozen P2 denominator or person allowlist is incomplete.")
    return _content_addressed(
        {
            "schema_version": P2_SCHEMA,
            "status": "ready_for_governed_context_workflow",
            "p0_content_sha256": p0_content_sha256,
            "p0_receipt_content_sha256": p0["receipt_content_sha256"],
            "cohort_sha256": manifest["evaluation_cohort"]["cohort_sha256"],
            "binding_set_sha256": manifest["canonical_bindings"][
                "binding_set_sha256"
            ],
            "canonical_person_allowlist_sha256": _hash(people),
            "recording_count": len(selected),
            "speaker_slot_count": sum(len(item["speaker_labels"]) for item in selected),
            "context_status_counts": dict(
                sorted(Counter(item["context_status"] for item in selected).items())
            ),
            "workflow": {
                "phases": ["clue_discovery", "identity_evaluation"],
                "host_validates_structured_output": True,
                "max_reference_repair_cycles_per_phase": 1,
                "persistent_session_transport": "stdio",
                "provider_retrieval_is_host_owned": True,
            },
            "action_counts": dict(ACTION_COUNTS),
            "will_read_gold": False,
            "will_apply_speaker_identity": False,
            "will_perform_external_provider_write": False,
        }
    )


def _proposal_slot_rows(
    *,
    document_id: str,
    speaker_labels: Sequence[str],
    prediction: Mapping[str, Any],
    canonical_people: set[str],
) -> list[dict[str, Any]]:
    proposals = [
        item for item in prediction.get("proposals") or [] if isinstance(item, Mapping)
    ]
    rows = []
    for label in speaker_labels:
        matches = [
            item
            for item in proposals
            if label in [str(value) for value in item.get("speaker_labels") or []]
        ]
        candidates = []
        for item in matches:
            person_id = str(item.get("person_id") or "")
            status = str(item.get("status") or "")
            confidence = item.get("confidence") if isinstance(item.get("confidence"), Mapping) else {}
            allowed = status == "candidate_match" and person_id in canonical_people
            candidates.append(
                {
                    "proposal_id": str(item.get("proposal_id") or ""),
                    "status": status,
                    "person_id": person_id if allowed else None,
                    "prepared_person_id": person_id if person_id in canonical_people else None,
                    "confidence_band": str(confidence.get("band") or "none"),
                    "confidence_numeric": (
                        int(confidence["numeric"])
                        if isinstance(confidence.get("numeric"), int)
                        else None
                    ),
                    "transcript_clue_ids": [
                        str(value) for value in item.get("transcript_clue_ids") or []
                    ],
                    "provenance_source_ids": [
                        str(value) for value in item.get("provenance_source_ids") or []
                    ],
                    "review_flags": [str(value) for value in item.get("review_flags") or []],
                    "factors": list(item.get("factors") or []),
                }
            )
        eligible = [item for item in candidates if item["person_id"]]
        if len({item["person_id"] for item in eligible}) == 1:
            disposition = "candidate"
            reason = "one_prepared_context_candidate"
            candidate_person_id = eligible[0]["person_id"]
        elif len({item["person_id"] for item in eligible}) > 1:
            disposition = "review"
            reason = "multiple_prepared_context_candidates"
            candidate_person_id = None
        elif matches:
            disposition = "abstain"
            reason = "no_prepared_candidate_match"
            candidate_person_id = None
        else:
            disposition = "unavailable"
            reason = "speaker_missing_from_context_evaluation"
            candidate_person_id = None
        rows.append(
            {
                "speaker_ref": f"{document_id}::{label}",
                "speaker_label": label,
                "disposition": disposition,
                "reason_code": reason,
                "candidate_person_id": candidate_person_id,
                "candidates": candidates,
            }
        )
    return rows


def _failure_case(
    *, document_id: str, speaker_labels: Sequence[str], stage: str,
    message: str, run_references: Mapping[str, Any],
) -> dict[str, Any]:
    return _content_addressed(
        {
            "schema_version": CASE_SCHEMA,
            "status": "context_workflow_unavailable",
            "document_id": document_id,
            "failure_stage": stage,
            "failure_detail": str(message)[:500],
            "run_references": dict(run_references),
            "speaker_slots": [
                {
                    "speaker_ref": f"{document_id}::{label}",
                    "speaker_label": label,
                    "disposition": "unavailable",
                    "reason_code": "context_workflow_failed",
                    "candidate_person_id": None,
                    "candidates": [],
                }
                for label in speaker_labels
            ],
            "provider_failures": [],
            "warnings": [],
            "action_counts": dict(ACTION_COUNTS),
        }
    )


def _successful_case(
    *, document_id: str, speaker_labels: Sequence[str], result: Mapping[str, Any],
    canonical_people: set[str],
) -> dict[str, Any]:
    prediction = result.get("prediction")
    if not isinstance(prediction, Mapping):
        raise Plan0064P2Error("The contextual workflow returned no prediction object.")
    rows = _proposal_slot_rows(
        document_id=document_id,
        speaker_labels=speaker_labels,
        prediction=prediction,
        canonical_people=canonical_people,
    )
    return _content_addressed(
        {
            "schema_version": CASE_SCHEMA,
            "status": "context_workflow_complete",
            "document_id": document_id,
            "evaluation_id": str(prediction.get("evaluation_id") or ""),
            "evaluation_status": str(prediction.get("status") or ""),
            "run_references": dict(result.get("run_references") or {}),
            "execution_provider": str(
                result.get("execution_provider") or "codex-app-server"
            ),
            "speaker_slots": rows,
            "prediction": dict(prediction),
            "provider_failures": list(prediction.get("source_failures") or []),
            "warnings": [str(value) for value in prediction.get("warnings") or []],
            "action_counts": dict(ACTION_COUNTS),
        }
    )


def execute_p2(
    p0_content_sha256: str,
    *,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    runner_factory: Callable[[], Callable[[str], Mapping[str, Any]]] = LocalSpeakerCaseRunner,
) -> dict[str, Any]:
    """Run missing cases once, checkpoint each, and freeze the complete P2 receipt."""
    preview = build_p2_preview(p0_content_sha256, runtime_root=runtime_root)
    p0, hydration_bridge = _phase_safe_p0(
        p0_content_sha256, runtime_root=runtime_root
    )
    manifest = _read_object(Path(p0["private_manifest_path"]))
    selected = [item for item in manifest["evaluation_cohort"]["considered"]
                if item["disposition"] == "selected_evaluation_candidate"]
    canonical_people = {
        item["person_id"]
        for item in manifest["canonical_bindings"]["subject_bindings"]
        if item["identity_candidate_eligible"]
    }
    root = runtime_root.expanduser().absolute() / f"p2-{preview['content_sha256'][:24]}"
    cases_root = root / "cases"
    receipt_path = root / "receipt.json"
    ensure_private_tree(root, cases_root)
    bridge_path = root / "identity-hydration-bridge.json"
    if hydration_bridge is not None:
        if bridge_path.exists():
            require_private_file(bridge_path, root)
            if _read_object(bridge_path) != hydration_bridge:
                raise Plan0064P2Error("The P0 identity-hydration bridge conflicts.")
        else:
            write_immutable_private_json(bridge_path, hydration_bridge)
    runner = runner_factory()
    cases = []
    for index, recording in enumerate(selected, start=1):
        document_id = str(recording["document_id"])
        case_path = cases_root / f"{document_id}.json"
        if case_path.exists():
            require_private_file(case_path, root)
            case = _read_object(case_path)
            if case.get("content_sha256") != _hash(
                {key: value for key, value in case.items() if key != "content_sha256"}
            ):
                raise Plan0064P2Error("A checkpointed context case drifted.")
        else:
            print(f"Running Plan 0064 P2 context case {index}/{len(selected)}...", flush=True)
            try:
                result = runner(document_id)
                case = _successful_case(
                    document_id=document_id,
                    speaker_labels=recording["speaker_labels"],
                    result=result,
                    canonical_people=canonical_people,
                )
            except CasePredictionFailure as exc:
                case = _failure_case(
                    document_id=document_id,
                    speaker_labels=recording["speaker_labels"],
                    stage=exc.stage,
                    message=str(exc),
                    run_references=exc.run_references,
                )
            write_immutable_private_json(case_path, case)
        if [row["speaker_label"] for row in case["speaker_slots"]] != list(
            recording["speaker_labels"]
        ):
            raise Plan0064P2Error("A context case changed the frozen slot denominator.")
        cases.append(case)
    slots = [row for case in cases for row in case["speaker_slots"]]
    summary = {
        "recording_count": len(cases),
        "speaker_slot_count": len(slots),
        "successful_case_count": sum(case["status"] == "context_workflow_complete" for case in cases),
        "unavailable_case_count": sum(case["status"] != "context_workflow_complete" for case in cases),
        "disposition_counts": dict(sorted(Counter(row["disposition"] for row in slots).items())),
        "reason_code_counts": dict(sorted(Counter(row["reason_code"] for row in slots).items())),
        "provider_failure_count": sum(len(case["provider_failures"]) for case in cases),
        "warning_count": sum(len(case["warnings"]) for case in cases),
        "app_intelligence_run_count": sum(len(case["run_references"]) for case in cases),
        "execution_provider_counts": dict(
            sorted(
                Counter(
                    str(case.get("execution_provider") or "unavailable")
                    for case in cases
                ).items()
            )
        ),
    }
    case_hashes = [
        case["content_sha256"]
        for case in sorted(cases, key=lambda item: item["document_id"])
    ]
    receipt = _content_addressed(
        {
            "schema_version": RECEIPT_SCHEMA,
            "status": "p2_complete_zero_identity_effect",
            "preview_content_sha256": preview["content_sha256"],
            "p0_content_sha256": p0_content_sha256,
            "case_content_sha256s": case_hashes,
            "case_set_sha256": _hash(case_hashes),
            "identity_hydration_bridge_content_sha256": (
                hydration_bridge["content_sha256"] if hydration_bridge else None
            ),
            "summary": summary,
            "action_counts": dict(ACTION_COUNTS),
        }
    )
    if receipt_path.exists():
        require_private_file(receipt_path, root)
        if _read_object(receipt_path) != receipt:
            raise Plan0064P2Error("The P2 terminal receipt conflicts with current cases.")
    else:
        write_immutable_private_json(receipt_path, receipt)
    return {**receipt, "private_cases_root": str(cases_root),
            "private_receipt_path": str(receipt_path), "idempotent_replay": False}


def replay_p2(
    p0_content_sha256: str,
    *,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
) -> dict[str, Any]:
    preview = build_p2_preview(p0_content_sha256, runtime_root=runtime_root)
    _p0, hydration_bridge = _phase_safe_p0(
        p0_content_sha256, runtime_root=runtime_root
    )
    root = runtime_root.expanduser().absolute() / f"p2-{preview['content_sha256'][:24]}"
    receipt_path = root / "receipt.json"
    require_private_file(receipt_path, root)
    receipt = _read_object(receipt_path)
    bridge_path = root / "identity-hydration-bridge.json"
    if hydration_bridge is not None:
        require_private_file(bridge_path, root)
        if _read_object(bridge_path) != hydration_bridge:
            raise Plan0064P2Error("The P0 hydration bridge drifted on replay.")
    case_hashes = []
    for path in sorted((root / "cases").glob("*.json")):
        require_private_file(path, root)
        case = _read_object(path)
        if case.get("content_sha256") != _hash(
            {key: value for key, value in case.items() if key != "content_sha256"}
        ):
            raise Plan0064P2Error("A P2 context case drifted on replay.")
        case_hashes.append(case["content_sha256"])
    if (
        len(case_hashes) != preview["recording_count"]
        or receipt.get("preview_content_sha256") != preview["content_sha256"]
        or receipt.get("case_content_sha256s") != case_hashes
        or receipt.get("case_set_sha256") != _hash(case_hashes)
        or receipt.get("identity_hydration_bridge_content_sha256")
        != (hydration_bridge["content_sha256"] if hydration_bridge else None)
        or receipt.get("content_sha256") != _hash(
            {key: value for key, value in receipt.items() if key != "content_sha256"}
        )
        or receipt.get("action_counts") != ACTION_COUNTS
    ):
        raise Plan0064P2Error("The frozen P2 receipt drifted.")
    return {**receipt, "private_cases_root": str(root / "cases"),
            "private_receipt_path": str(receipt_path), "idempotent_replay": True}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preview", "execute", "replay"))
    parser.add_argument("--p0-content-sha256", required=True)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument(
        "--runner",
        choices=("codex-app-server", "openai-compatible"),
        default="codex-app-server",
    )
    args = parser.parse_args(argv)
    if args.action == "preview":
        result = build_p2_preview(args.p0_content_sha256, runtime_root=args.runtime_root)
    elif args.action == "execute":
        factory = (
            OpenAICompatibleCaseRunner
            if args.runner == "openai-compatible"
            else LocalSpeakerCaseRunner
        )
        result = execute_p2(
            args.p0_content_sha256,
            runtime_root=args.runtime_root,
            runner_factory=factory,
        )
    else:
        result = replay_p2(args.p0_content_sha256, runtime_root=args.runtime_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
