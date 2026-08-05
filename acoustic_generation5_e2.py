"""Plan 0055 E2 gold-blind paired prediction and acoustic matrices."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import struct
import subprocess
import urllib.error
import urllib.request
import wave
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import acoustic_audio_derivatives as p1
import acoustic_generation3_recalibration as recalibration
import acoustic_generation4_acoustic_contract as acoustic_contract
import acoustic_generation5_evaluation_gold as evaluation_gold
import acoustic_generation5_source_j1_acceptance as j1
import acoustic_generation5_source_review as s1
import acoustic_speech_preparation as p2
import acoustic_verification as verification
from acoustic_audio_derivatives import (
    ensure_private_tree,
    require_private_file,
    sha256_file,
    write_immutable_private_json,
)


PREVIEW_SCHEMA = "transcribe-audio.generation5-e2-preview.v1"
MANIFEST_SCHEMA = "transcribe-audio.generation5-e2-manifest.v1"
RECEIPT_SCHEMA = "transcribe-audio.generation5-e2-receipt.v1"
EXECUTION_SCHEMA = "transcribe-audio.generation5-e2-execution.v1"
REPLAY_SCHEMA = "transcribe-audio.generation5-e2-replay.v1"
MATRIX_SCHEMA = "transcribe-audio.generation5-e2-acoustic-matrix.v1"
WORKER_PACKET_SCHEMA = "transcribe-audio.generation5-e2-worker-packet.v1"
PREDICTION_SCHEMA = "transcribe-audio.generation5-e2-predictions.v1"
J1_PREVIEW_SHA256 = "b0c642d5989df72e876abbbf10427148e72c1cf3b2c8fac69eaf90e5062ff3a3"
J1_MANIFEST_SHA256 = "617b98be57f28770e1b22ecaaf29568518806c73b0906c4c3abd1f84493c0aac"
S1_PREVIEW_SHA256 = "5a3f9fc9848a5e0b669bc37796e5a55b4f9dcd7bf0f55609aefa886e4caabcf9"
S1_MANIFEST_SHA256 = "04a860f3823c82f513dc655970fbe9b9d99641cec3f343d2e055095f39bb9a84"
THRESHOLD_APPLICATION = Path(
    "~/.local/state/transcribe-audio/plan-0037/generation-3/"
    "recalibration-executions/generation3-recalibration-execution-"
    "39298c74aab4a773945268cd/threshold-application.json"
)
THRESHOLD_APPLICATION_SHA256 = "308f326d3fe9baa175ed32c90df4255a8d4bfc1924c6f925eab490ae2832f4d1"
DEFAULT_RUNTIME_ROOT = Path("~/.local/state/transcribe-audio/plan-0055/e2")
DEFAULT_PROVIDER = "openrouter"
DEFAULT_MODEL = "openai/gpt-5.2"
DEFAULT_WORKER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
MODULE_NAME = Path(__file__).name
SELECTED_ORDINALS = (1, 2, 3, 4, 5, 6, 7)
EXPECTED_SPEAKER_COUNT = 22
EXPECTED_MATRIX_COUNT = 9
EXPECTED_TRIAL_COUNT = 396
PRESERVED_CONTEXT_PREDICTION = Path(
    "~/.local/state/transcribe-audio/plan-0055/e2/"
    "generation5-e2-aeb87c4527f1bc2f2fff5112/context-predictions.json"
)
PRESERVED_CONTEXT_PREDICTION_FILE_SHA256 = "7624869095f4d2473854c20ed51da22914a7e80256c201352dc0457740c0384e"
PRESERVED_CONTEXT_PREDICTION_CONTENT_SHA256 = "3bea4134ab9ebc67970c1266e9ce98d648f1453624a0c196d4a9e0b7161740d4"
FORBIDDEN_WORKER_KEYS = {
    "answer_sha256", "enrolled_subject_id", "identity_authority", "person_id",
    "population", "population_result", "private_gold", "private_identity_display",
    "speaker_gold",
}


class Generation5E2Error(ValueError):
    """Raised when E2 cannot remain complete, private, and gold-blind."""


def _canonical_hash(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Generation5E2Error("Private E2 authority is unreadable.") from exc
    if not isinstance(value, dict):
        raise Generation5E2Error("Private E2 authority must be an object.")
    return value


def _git(arguments: Sequence[str], *, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", *arguments], cwd=Path(__file__).resolve().parent,
        capture_output=True, text=not binary, check=False,
    )
    if result.returncode:
        raise Generation5E2Error("Repository authority is unavailable.")
    return result.stdout if binary else result.stdout.strip()


def _repository_authority() -> dict[str, Any]:
    if _git(["status", "--porcelain=v1", "--untracked-files=normal"]):
        raise Generation5E2Error("Repository must be clean.")
    if str(_git(["rev-list", "--left-right", "--count", "@{upstream}...HEAD"])).split() != ["0", "0"]:
        raise Generation5E2Error("Repository must be upstream-even.")
    commit = str(_git(["rev-parse", "HEAD"]))
    body = _git(["show", f"{commit}:{MODULE_NAME}"], binary=True)
    if not isinstance(body, bytes) or hashlib.sha256(body).hexdigest() != sha256_file(Path(__file__).resolve()):
        raise Generation5E2Error("Committed E2 module drifted.")
    return {"commit": commit, "module_sha256": hashlib.sha256(body).hexdigest(),
            "clean": True, "upstream_ahead": 0, "upstream_behind": 0}


def _reject_forbidden_worker_content(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).casefold() in FORBIDDEN_WORKER_KEYS:
                raise Generation5E2Error("A worker packet contains sealed gold metadata.")
            _reject_forbidden_worker_content(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            _reject_forbidden_worker_content(child)


def build_context_worker_packet(cards: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build the exact transcript-only packet without reading identity labels."""
    speakers = []
    seen: set[str] = set()
    for raw in cards:
        card = dict(raw)
        reference = str(card.get("speaker_ref") or "")
        snippets = card.get("clip", {}).get("snippets") if isinstance(card.get("clip"), Mapping) else None
        if (
            not reference or reference in seen
            or int(card.get("ordinal") or 0) not in SELECTED_ORDINALS
            or not isinstance(snippets, list) or not snippets
        ):
            raise Generation5E2Error("Blinded context-card inventory is invalid.")
        seen.add(reference)
        speakers.append({
            "speaker_ref": reference,
            "recording": str(card.get("display_case") or ""),
            "speaker_label": str(card.get("speaker_label") or ""),
            "transcript_clues": [
                " ".join(str(item.get("text") or "").split())
                for item in snippets if isinstance(item, Mapping) and str(item.get("text") or "").strip()
            ],
        })
    if len(speakers) != EXPECTED_SPEAKER_COUNT or len(seen) != EXPECTED_SPEAKER_COUNT:
        raise Generation5E2Error("Exactly 22 blinded speaker cards are required.")
    core = {
        "schema_version": WORKER_PACKET_SCHEMA,
        "worker_lane": "context_only",
        "instructions": (
            "Identify each speaker from transcript context only. Use a real name when grounded, "
            "otherwise a stable descriptive alias. Do not invent certainty. Return every speaker."
        ),
        "allowed_dispositions": ["assign", "review", "abstain"],
        "allowed_confidence_bands": ["high", "medium", "low", "none"],
        "speakers": speakers,
        "speaker_count": len(speakers),
        "contains_acoustic_evidence": False,
        "contains_gold": False,
        "contains_competing_worker_output": False,
    }
    _reject_forbidden_worker_content(core)
    return {**core, "content_sha256": _canonical_hash(core)}


def build_augmented_worker_packet(
    context_packet: Mapping[str, Any], matrices: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Add separately visible voice evidence without exposing another worker."""
    if context_packet.get("worker_lane") != "context_only" or len(matrices) != EXPECTED_MATRIX_COUNT:
        raise Generation5E2Error("Augmented worker requires all nine acoustic matrices.")
    matrix_ids = {(str(item.get("candidate_id")), str(item.get("method_id"))) for item in matrices}
    expected = {(candidate, method) for candidate in acoustic_contract.CANDIDATE_IDS
                for method in acoustic_contract.METHOD_IDS}
    if matrix_ids != expected or any(item.get("speaker_count") != EXPECTED_SPEAKER_COUNT for item in matrices):
        raise Generation5E2Error("All nine complete acoustic matrices are required.")
    core = {
        "schema_version": WORKER_PACKET_SCHEMA,
        "worker_lane": "voice_augmented",
        "instructions": (
            "Identify every speaker using the unchanged transcript context plus the separately "
            "labeled acoustic evidence. Acoustic scores only compare against the two enrolled "
            "people and may be uncertain or conflicting. Do not treat a weak score as identity proof."
        ),
        "allowed_dispositions": list(context_packet["allowed_dispositions"]),
        "allowed_confidence_bands": list(context_packet["allowed_confidence_bands"]),
        "speakers": list(context_packet["speakers"]),
        "speaker_count": context_packet["speaker_count"],
        "acoustic_matrices": [dict(item) for item in matrices],
        "acoustic_matrix_count": len(matrices),
        "contains_acoustic_evidence": True,
        "contains_gold": False,
        "contains_competing_worker_output": False,
    }
    _reject_forbidden_worker_content(core)
    return {**core, "content_sha256": _canonical_hash(core)}


def _prediction_json_schema(expected_refs: Sequence[str]) -> dict[str, Any]:
    return {
        "type": "object", "additionalProperties": False,
        "required": ["predictions"],
        "properties": {"predictions": {
            "type": "array", "minItems": len(expected_refs), "maxItems": len(expected_refs),
            "items": {"type": "object", "additionalProperties": False,
                      "required": ["speaker_ref", "identity_or_alias", "confidence_band", "disposition", "rationale"],
                      "properties": {
                          "speaker_ref": {"type": "string", "enum": list(expected_refs)},
                          "identity_or_alias": {"type": "string", "minLength": 1, "maxLength": 200},
                          "confidence_band": {"type": "string", "enum": ["high", "medium", "low", "none"]},
                          "disposition": {"type": "string", "enum": ["assign", "review", "abstain"]},
                          "rationale": {"type": "string", "minLength": 1, "maxLength": 500},
                      }},
        }},
    }


def validate_predictions(
    value: Mapping[str, Any], *, expected_refs: Sequence[str], worker_lane: str,
) -> dict[str, Any]:
    predictions = value.get("predictions")
    if not isinstance(predictions, list) or len(predictions) != len(expected_refs):
        raise Generation5E2Error("Worker prediction denominator is incomplete.")
    by_ref: dict[str, dict[str, Any]] = {}
    for raw in predictions:
        if not isinstance(raw, Mapping):
            raise Generation5E2Error("A worker prediction is invalid.")
        item = dict(raw)
        ref = str(item.get("speaker_ref") or "")
        if (
            ref not in expected_refs or ref in by_ref
            or set(item) != {"speaker_ref", "identity_or_alias", "confidence_band", "disposition", "rationale"}
            or item.get("confidence_band") not in {"high", "medium", "low", "none"}
            or item.get("disposition") not in {"assign", "review", "abstain"}
            or not " ".join(str(item.get("identity_or_alias") or "").split())
            or not " ".join(str(item.get("rationale") or "").split())
        ):
            raise Generation5E2Error("A worker prediction is invalid or duplicated.")
        by_ref[ref] = item
    if set(by_ref) != set(expected_refs):
        raise Generation5E2Error("Worker prediction references are incomplete.")
    core = {"schema_version": PREDICTION_SCHEMA, "worker_lane": worker_lane,
            "speaker_count": len(expected_refs),
            "predictions": [by_ref[ref] for ref in expected_refs],
            "contains_gold": False, "contains_competing_worker_output": False}
    return {**core, "content_sha256": _canonical_hash(core)}


def _paths(runtime_root: Path, content_sha256: str) -> dict[str, Path]:
    root = runtime_root.expanduser().absolute()
    run = root / f"generation5-e2-{content_sha256[:24]}"
    return {"root": root, "run": run, "manifest": run / "private-manifest.json",
            "receipt": run / "receipt.json", "execution": run / "execution.json",
            "context_predictions": run / "context-predictions.json",
            "augmented_predictions": run / "augmented-predictions.json",
            "matrices": run / "matrices", "p1": run / "p1", "p2": run / "p2"}


def _frozen_inputs() -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    replay = j1.replay_generation5_source_j1_acceptance(J1_PREVIEW_SHA256)
    j1_paths = j1._paths(j1.DEFAULT_RUNTIME_ROOT, J1_PREVIEW_SHA256)
    if replay.get("idempotent_replay") is not True or sha256_file(j1_paths["manifest"]) != J1_MANIFEST_SHA256:
        raise Generation5E2Error("Frozen J1 authority drifted.")
    j1_preview = _read_json(j1_paths["manifest"]).get("preview")
    selected = j1_preview.get("private_gold", {}).get("selected_cases") if isinstance(j1_preview, Mapping) else None
    if not isinstance(selected, list) or [int(item.get("enumerated_ordinal") or 0) for item in selected] != list(SELECTED_ORDINALS):
        raise Generation5E2Error("Frozen seven-recording membership drifted.")

    s1_replay = s1.replay_generation5_source_review(S1_PREVIEW_SHA256)
    s1_paths = s1._paths(s1.DEFAULT_RUNTIME_ROOT, S1_PREVIEW_SHA256)
    if s1_replay.get("idempotent_replay") is not True or sha256_file(s1_paths["manifest"]) != S1_MANIFEST_SHA256:
        raise Generation5E2Error("Frozen S1 review authority drifted.")
    s1_preview = _read_json(s1_paths["manifest"]).get("preview")
    evidence = s1_preview.get("private_evidence") if isinstance(s1_preview, Mapping) else None
    cards = evidence.get("cards") if isinstance(evidence, Mapping) else None
    if not isinstance(cards, list):
        raise Generation5E2Error("Frozen S1 cards are unavailable.")
    selected_cards = [dict(card) for card in cards if isinstance(card, Mapping)
                      and int(card.get("ordinal") or 0) in SELECTED_ORDINALS]
    packet = build_context_worker_packet(selected_cards)
    return dict(j1_preview), selected_cards, packet


def _threshold_authority() -> dict[str, Any]:
    path = THRESHOLD_APPLICATION.expanduser().absolute()
    require_private_file(path, path.parents[2])
    application = _read_json(path)
    units = application.get("thresholds")
    observed = {(str(item.get("candidate_id")), str(item.get("method_id")))
                for item in units or [] if isinstance(item, Mapping)}
    expected = {(candidate, method) for candidate in acoustic_contract.CANDIDATE_IDS
                for method in acoustic_contract.METHOD_IDS}
    if (
        sha256_file(path) != THRESHOLD_APPLICATION_SHA256
        or application.get("status") != "success"
        or application.get("threshold_unit_count") != EXPECTED_MATRIX_COUNT
        or observed != expected
    ):
        raise Generation5E2Error("Nine-unit calibration threshold authority drifted.")
    return application


def preview_generation5_e2(
    *, repository_authority: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    j1_preview, cards, context_packet = _frozen_inputs()
    thresholds = _threshold_authority()
    clip_bindings = []
    s1_paths = s1._paths(s1.DEFAULT_RUNTIME_ROOT, S1_PREVIEW_SHA256)
    for card in cards:
        clip = s1_paths["clips"] / f"{s1._slug(card)}.wav"
        require_private_file(clip, s1_paths["root"])
        clip_bindings.append({
            "speaker_ref": card["speaker_ref"], "ordinal": card["ordinal"],
            "speaker_label": card["speaker_label"], "clip_path": str(clip),
            "clip_sha256": sha256_file(clip),
        })
    if len({item["clip_sha256"] for item in clip_bindings}) != EXPECTED_SPEAKER_COUNT:
        raise Generation5E2Error("Every selected speaker requires a distinct bound clip.")
    actions = {
        "run_context_only_worker": True,
        "run_nine_acoustic_matrices": True,
        "run_voice_augmented_worker": True,
        "reveal_gold": False,
        "score_against_gold": False,
        "mutate_profiles_or_references": False,
        "enable_default_integration": False,
        "run_historical_reprocessing": False,
    }
    private = {
        "context_worker_packet": context_packet,
        "clip_bindings": clip_bindings,
        "threshold_units": [
            {key: item[key] for key in ("candidate_id", "method_id", "threshold", "temperature")}
            for item in thresholds["thresholds"]
        ],
    }
    core = {
        "schema_version": PREVIEW_SCHEMA,
        "status": "gold_blind_workers_ready",
        "repository_authority": dict(repository_authority or _repository_authority()),
        "j1_preview_sha256": J1_PREVIEW_SHA256,
        "j1_manifest_sha256": J1_MANIFEST_SHA256,
        "selected_case_ids_sha256": j1_preview["selected_case_ids_sha256"],
        "selected_source_set_sha256": j1_preview["selected_source_set_sha256"],
        "selected_transcript_set_sha256": j1_preview["selected_transcript_set_sha256"],
        "speaker_count": EXPECTED_SPEAKER_COUNT,
        "acoustic_matrix_count": EXPECTED_MATRIX_COUNT,
        "acoustic_trial_count": EXPECTED_TRIAL_COUNT,
        "context_worker_packet_sha256": context_packet["content_sha256"],
        "threshold_application_sha256": THRESHOLD_APPLICATION_SHA256,
        "threshold_unit_set_sha256": _canonical_hash(private["threshold_units"]),
        "worker_runtime": {
            "provider": DEFAULT_PROVIDER, "model": DEFAULT_MODEL,
            "endpoint_sha256": hashlib.sha256(DEFAULT_WORKER_ENDPOINT.encode()).hexdigest(),
            "tools_enabled": False, "provider_storage_requested": False,
            "provider_fallbacks_allowed": False, "structured_output_required": True,
        },
        "superseded_no_output_attempt": {
            "provider": "openai", "model": "gpt-5.2", "attempt_count": 1,
            "status": "failed_no_output", "reason_code": "http_429",
            "prediction_captured": False, "acoustic_models_ran": False,
            "gold_revealed": False,
        },
        "preserved_context_prediction": {
            "content_sha256": PRESERVED_CONTEXT_PREDICTION_CONTENT_SHA256,
            "file_sha256": PRESERVED_CONTEXT_PREDICTION_FILE_SHA256,
            "speaker_count": EXPECTED_SPEAKER_COUNT, "worker_lane": "context_only",
            "source_authority_sha256": "aeb87c4527f1bc2f2fff51129d33e32ee5b84bd5a01d22660b2c489b882d2d3f",
            "reuse_exact_prediction": True, "rerun_context_worker": False,
            "subsequent_failure_reason": "enrolled_identity_helper_reference_error",
            "acoustic_models_ran_after_prediction": False, "gold_revealed": False,
        },
        "private_evidence": private,
        "action_vector": actions,
        "contains_private_paths": True,
        "contains_gold": False,
        "did_reveal_gold": False,
        "did_run_workers_or_models": False,
        "did_mutate_profiles_or_references": False,
    }
    _reject_forbidden_worker_content(private)
    return {**core, "content_sha256": _canonical_hash(core)}


def _portable(preview: Mapping[str, Any]) -> dict[str, Any]:
    return {key: preview[key] for key in (
        "status", "j1_preview_sha256", "j1_manifest_sha256",
        "selected_case_ids_sha256", "selected_source_set_sha256",
        "selected_transcript_set_sha256", "speaker_count", "acoustic_matrix_count",
        "acoustic_trial_count", "context_worker_packet_sha256",
        "threshold_application_sha256", "threshold_unit_set_sha256", "action_vector",
        "worker_runtime", "superseded_no_output_attempt",
        "preserved_context_prediction",
    )}


def apply_generation5_e2(
    reviewed_preview: Mapping[str, Any], *, expected_content_sha256: str,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
) -> dict[str, Any]:
    preview = preview_generation5_e2()
    if dict(reviewed_preview) != preview or preview["content_sha256"] != expected_content_sha256:
        raise Generation5E2Error("Reviewed E2 preview is stale.")
    paths = _paths(runtime_root, expected_content_sha256)
    if paths["receipt"].exists():
        return replay_generation5_e2(expected_content_sha256, runtime_root=runtime_root)
    ensure_private_tree(paths["root"], paths["run"])
    manifest = {"schema_version": MANIFEST_SCHEMA, "status": "frozen_gold_blind_inputs",
                "preview": preview}
    write_immutable_private_json(paths["manifest"], manifest)
    receipt = {**_portable(preview), "schema_version": RECEIPT_SCHEMA,
               "preview_content_sha256": expected_content_sha256,
               "manifest_sha256": sha256_file(paths["manifest"]), "mode": "0600",
               "did_reveal_gold": False, "did_run_workers_or_models": False}
    write_immutable_private_json(paths["receipt"], receipt)
    return {**receipt, "idempotent_replay": False}


def replay_generation5_e2(
    expected_content_sha256: str, *, runtime_root: Path = DEFAULT_RUNTIME_ROOT,
) -> dict[str, Any]:
    paths = _paths(runtime_root, expected_content_sha256)
    require_private_file(paths["manifest"], paths["root"])
    require_private_file(paths["receipt"], paths["root"])
    manifest, receipt = _read_json(paths["manifest"]), _read_json(paths["receipt"])
    preview = manifest.get("preview")
    if not isinstance(preview, Mapping):
        raise Generation5E2Error("Frozen E2 preview is missing.")
    core = {key: value for key, value in preview.items() if key != "content_sha256"}
    expected_receipt = {**_portable(preview), "schema_version": RECEIPT_SCHEMA,
                        "preview_content_sha256": expected_content_sha256,
                        "manifest_sha256": sha256_file(paths["manifest"]), "mode": "0600",
                        "did_reveal_gold": False, "did_run_workers_or_models": False}
    if (_canonical_hash(core) != expected_content_sha256
            or preview.get("content_sha256") != expected_content_sha256
            or manifest != {"schema_version": MANIFEST_SCHEMA, "status": "frozen_gold_blind_inputs", "preview": preview}
            or receipt != expected_receipt):
        raise Generation5E2Error("Frozen E2 authority drifted.")
    return {**receipt, "replay_schema_version": REPLAY_SCHEMA, "idempotent_replay": True}


def _read_pcm16_mono(path: Path) -> tuple[float, ...]:
    try:
        with wave.open(str(path), "rb") as reader:
            if (reader.getnchannels(), reader.getsampwidth(), reader.getframerate(), reader.getcomptype()) != (1, 2, 16_000, "NONE"):
                raise Generation5E2Error("E2 acoustic PCM contract is invalid.")
            frames = reader.readframes(reader.getnframes())
    except (EOFError, OSError, wave.Error) as exc:
        raise Generation5E2Error("E2 acoustic PCM is unreadable.") from exc
    if not frames or len(frames) % 2:
        raise Generation5E2Error("E2 acoustic PCM is empty or truncated.")
    return tuple(value / 32768.0 for value in struct.unpack(f"<{len(frames) // 2}h", frames))


def _prepare_clip_methods(
    binding: Mapping[str, Any], *, authority_sha256: str, paths: Mapping[str, Path],
) -> dict[str, Path]:
    clip = Path(str(binding["clip_path"]))
    require_private_file(clip, clip.parents[2])
    if sha256_file(clip) != binding["clip_sha256"]:
        raise Generation5E2Error("A frozen E2 speaker clip drifted.")
    source_blob_id = "e2-source-" + str(binding["clip_sha256"])[:24]
    plan = p1.dry_run(
        clip, runtime_root=paths["p1"], source_blob_id=source_blob_id,
        expected_source_sha256=str(binding["clip_sha256"]),
        channel_policy="stereo_average_to_mono",
        channel_policy_authority_sha256=authority_sha256,
    )
    p1.apply_derivative(
        clip, runtime_root=paths["p1"], approval_token=p1.APPLY_TOKEN,
        source_blob_id=source_blob_id, expected_source_sha256=str(binding["clip_sha256"]),
        channel_policy="stereo_average_to_mono",
        channel_policy_authority_sha256=authority_sha256,
    )
    p1.replay_derivative(plan["run_id"], runtime_root=paths["p1"])
    p2_plan = p2.dry_run(
        plan["run_id"], p1_runtime_root=paths["p1"], runtime_root=paths["p2"],
        intended_split="evaluation", split_access_authority_sha256=authority_sha256,
    )
    applied = p2.apply_comparison(
        plan["run_id"], p1_runtime_root=paths["p1"], runtime_root=paths["p2"],
        intended_split="evaluation", split_access_authority_sha256=authority_sha256,
    )
    p2.replay_comparison(p2_plan["run_id"], runtime_root=paths["p2"])
    methods: dict[str, Path] = {}
    comparison = applied.get("comparison")
    for raw in comparison.get("method_results") or [] if isinstance(comparison, Mapping) else []:
        if not isinstance(raw, Mapping) or raw.get("method_id") not in acoustic_contract.METHOD_IDS:
            continue
        if raw.get("status") != "success":
            raise Generation5E2Error("A required E2 preparation method failed.")
        output = Path(str(raw.get("output_path") or ""))
        require_private_file(output, paths["root"])
        if sha256_file(output) != raw.get("output_sha256"):
            raise Generation5E2Error("An E2 preparation output drifted.")
        methods[str(raw["method_id"])] = output
    if set(methods) != set(acoustic_contract.METHOD_IDS):
        raise Generation5E2Error("Every E2 clip requires three score preparations.")
    return methods


def _profile_inventory() -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, str]]:
    profiles, _ = recalibration._active_profiles(
        calibration_root=recalibration.DEFAULT_CALIBRATION_ROOT,
        p3_runtime_root=recalibration.DEFAULT_P3_RUNTIME_ROOT,
    )
    adapters = verification.adapter_registry()
    enrolled = evaluation_gold._enrolled_identity_map()
    subject_to_name = {subject: name.title() for name, subject in enrolled.items()}
    subject_to_name.update({enrolled.get("chris williams", ""): "Chris Williams",
                            enrolled.get("eric cochran", ""): "Eric Cochran"})
    subject_to_name.pop("", None)
    if (
        len(profiles) != 6 or len(subject_to_name) != 2
        or set(adapters) != set(acoustic_contract.CANDIDATE_IDS)
        or any(sum(profile["candidate_id"] == candidate for profile in profiles) != 2
               for candidate in acoustic_contract.CANDIDATE_IDS)
    ):
        raise Generation5E2Error("The enrolled acoustic profile inventory drifted.")
    return profiles, adapters, subject_to_name


def _score_matrices(
    preview: Mapping[str, Any], *, paths: Mapping[str, Path],
) -> list[dict[str, Any]]:
    profiles, adapters, subject_to_name = _profile_inventory()
    thresholds = {(item["candidate_id"], item["method_id"]): item
                  for item in preview["private_evidence"]["threshold_units"]}
    rows_by_unit = {(candidate, method): [] for candidate in acoustic_contract.CANDIDATE_IDS
                    for method in acoustic_contract.METHOD_IDS}
    for index, binding in enumerate(preview["private_evidence"]["clip_bindings"], start=1):
        print(f"Preparing and scoring E2 speaker {index}/{EXPECTED_SPEAKER_COUNT}...", flush=True)
        methods = _prepare_clip_methods(binding, authority_sha256=preview["content_sha256"], paths=paths)
        for method_id, pcm_path in sorted(methods.items()):
            samples = _read_pcm16_mono(pcm_path)
            for candidate_id in acoustic_contract.CANDIDATE_IDS:
                scores = []
                for profile in profiles:
                    if profile["candidate_id"] != candidate_id:
                        continue
                    scored = verification.score_profile(
                        profile["profile_id"], adapter=adapters[candidate_id],
                        probe_samples=samples, sample_rate=16_000,
                        runtime_root=recalibration.DEFAULT_CALIBRATION_ROOT,
                        p3_runtime_root=recalibration.DEFAULT_P3_RUNTIME_ROOT,
                    )
                    label = subject_to_name.get(profile["person_ref_id"])
                    if not label or not math.isfinite(float(scored["score"])):
                        raise Generation5E2Error("An acoustic score is unbound or nonfinite.")
                    scores.append({"reference_identity": label, "score": float(scored["score"]),
                                   "trial_id": scored["trial_id"], "probe_sha256": scored["probe_sha256"]})
                scores.sort(key=lambda item: item["reference_identity"])
                if len(scores) != 2:
                    raise Generation5E2Error("Each acoustic row requires two enrolled scores.")
                ranked = sorted(scores, key=lambda item: (item["score"], item["reference_identity"]), reverse=True)
                threshold = thresholds[(candidate_id, method_id)]
                rows_by_unit[(candidate_id, method_id)].append({
                    "speaker_ref": binding["speaker_ref"], "scores": scores,
                    "highest_reference_identity": ranked[0]["reference_identity"],
                    "highest_score": ranked[0]["score"],
                    "score_margin": ranked[0]["score"] - ranked[1]["score"],
                    "threshold_pass": ranked[0]["score"] >= float(threshold["threshold"]),
                })
    matrices = []
    ensure_private_tree(paths["root"], paths["matrices"])
    for candidate_id, method_id in sorted(rows_by_unit):
        rows = rows_by_unit[(candidate_id, method_id)]
        threshold = thresholds[(candidate_id, method_id)]
        core = {"schema_version": MATRIX_SCHEMA, "status": "complete",
                "candidate_id": candidate_id, "method_id": method_id,
                "threshold": threshold["threshold"], "temperature": threshold["temperature"],
                "speaker_count": len(rows), "profile_count": 2,
                "trial_count": len(rows) * 2, "rows": rows,
                "contains_gold": False, "did_select_or_change_thresholds": False}
        matrix = {**core, "content_sha256": _canonical_hash(core)}
        write_immutable_private_json(paths["matrices"] / f"{candidate_id}--{method_id}.json", matrix)
        matrices.append(matrix)
    if len(matrices) != EXPECTED_MATRIX_COUNT or sum(item["trial_count"] for item in matrices) != EXPECTED_TRIAL_COUNT:
        raise Generation5E2Error("The nine-matrix acoustic denominator is incomplete.")
    return matrices


def openai_worker(
    packet: Mapping[str, Any], *, model: str = DEFAULT_MODEL,
    api_key: str | None = None, timeout_seconds: float = 600,
) -> dict[str, Any]:
    """Run one no-tools, non-stored Responses API worker turn."""
    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise Generation5E2Error("OPENAI_API_KEY is unavailable for the isolated worker.")
    refs = [str(item["speaker_ref"]) for item in packet["speakers"]]
    body = {
        "model": model, "store": False,
        "reasoning": {"effort": "high"},
        "input": [{"role": "user", "content": [{"type": "input_text", "text": json.dumps(packet, ensure_ascii=False)}]}],
        "text": {"format": {"type": "json_schema", "name": "speaker_predictions",
                             "strict": True, "schema": _prediction_json_schema(refs)}},
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses", data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            result = json.loads(response.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise Generation5E2Error("The isolated prediction worker failed.") from exc
    output_text = ""
    for output in result.get("output") or []:
        for content in output.get("content") or [] if isinstance(output, Mapping) else []:
            if isinstance(content, Mapping) and content.get("type") == "output_text":
                output_text += str(content.get("text") or "")
    try:
        parsed = json.loads(output_text)
    except json.JSONDecodeError as exc:
        raise Generation5E2Error("The isolated worker returned invalid structured output.") from exc
    if not isinstance(parsed, dict):
        raise Generation5E2Error("The isolated worker output must be an object.")
    return parsed


def openrouter_worker(
    packet: Mapping[str, Any], *, model: str = DEFAULT_MODEL,
    api_key: str | None = None, timeout_seconds: float = 600,
) -> dict[str, Any]:
    """Run one exact-model, no-tools OpenRouter structured-output turn."""
    key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise Generation5E2Error("OPENROUTER_API_KEY is unavailable for the isolated worker.")
    refs = [str(item["speaker_ref"]) for item in packet["speakers"]]
    body = {
        "model": model,
        "messages": [{"role": "user", "content": json.dumps(packet, ensure_ascii=False)}],
        "reasoning": {"effort": "high"},
        "max_completion_tokens": 16_000,
        "provider": {"allow_fallbacks": False},
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "speaker_predictions", "strict": True,
            "schema": _prediction_json_schema(refs),
        }},
    }
    request = urllib.request.Request(
        DEFAULT_WORKER_ENDPOINT, data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            result = json.loads(response.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise Generation5E2Error("The isolated OpenRouter worker failed.") from exc
    try:
        content = result["choices"][0]["message"]["content"]
        parsed = json.loads(content)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise Generation5E2Error("The OpenRouter worker returned invalid structured output.") from exc
    if not isinstance(parsed, dict):
        raise Generation5E2Error("The isolated worker output must be an object.")
    return parsed


def execute_generation5_e2(
    expected_content_sha256: str, *, runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    worker: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Freeze both workers and all matrices before any gold reveal."""
    replay_generation5_e2(expected_content_sha256, runtime_root=runtime_root)
    paths = _paths(runtime_root, expected_content_sha256)
    if paths["execution"].exists():
        return replay_generation5_e2_execution(expected_content_sha256, runtime_root=runtime_root)
    preview = _read_json(paths["manifest"])["preview"]
    selected_worker = worker or openrouter_worker
    context_packet = preview["private_evidence"]["context_worker_packet"]
    expected_refs = [str(item["speaker_ref"]) for item in context_packet["speakers"]]

    if paths["context_predictions"].exists():
        require_private_file(paths["context_predictions"], paths["root"])
        context_predictions = _read_json(paths["context_predictions"])
    elif preview.get("preserved_context_prediction", {}).get("reuse_exact_prediction") is True:
        inherited_path = PRESERVED_CONTEXT_PREDICTION.expanduser().absolute()
        require_private_file(inherited_path, inherited_path.parents[2])
        inherited = _read_json(inherited_path)
        validated = validate_predictions(
            {"predictions": inherited.get("predictions")},
            expected_refs=expected_refs, worker_lane="context_only",
        )
        if (
            sha256_file(inherited_path) != PRESERVED_CONTEXT_PREDICTION_FILE_SHA256
            or inherited != validated
            or inherited.get("content_sha256") != PRESERVED_CONTEXT_PREDICTION_CONTENT_SHA256
        ):
            raise Generation5E2Error("The preserved context prediction drifted.")
        context_predictions = inherited
        write_immutable_private_json(paths["context_predictions"], context_predictions)
    else:
        print("Running isolated context-only prediction worker...", flush=True)
        context_predictions = validate_predictions(
            selected_worker(context_packet), expected_refs=expected_refs, worker_lane="context_only",
        )
        write_immutable_private_json(paths["context_predictions"], context_predictions)

    matrices = _score_matrices(preview, paths=paths)
    augmented_packet = build_augmented_worker_packet(context_packet, matrices)
    if paths["augmented_predictions"].exists():
        require_private_file(paths["augmented_predictions"], paths["root"])
        augmented_predictions = _read_json(paths["augmented_predictions"])
    else:
        print("Running isolated voice-augmented prediction worker...", flush=True)
        augmented_predictions = validate_predictions(
            selected_worker(augmented_packet), expected_refs=expected_refs, worker_lane="voice_augmented",
        )
        write_immutable_private_json(paths["augmented_predictions"], augmented_predictions)

    matrix_hashes = [item["content_sha256"] for item in matrices]
    core = {
        "schema_version": EXECUTION_SCHEMA,
        "status": "gold_blind_paired_predictions_frozen",
        "authority_content_sha256": expected_content_sha256,
        "worker_runtime": preview["worker_runtime"],
        "context_worker_packet_sha256": context_packet["content_sha256"],
        "context_predictions_sha256": context_predictions["content_sha256"],
        "augmented_worker_packet_sha256": augmented_packet["content_sha256"],
        "augmented_predictions_sha256": augmented_predictions["content_sha256"],
        "matrix_count": len(matrices), "matrix_set_sha256": _canonical_hash(matrix_hashes),
        "matrix_content_sha256s": matrix_hashes,
        "speaker_count": EXPECTED_SPEAKER_COUNT,
        "trial_count": sum(item["trial_count"] for item in matrices),
        "private_evidence": {
            "context_predictions": context_predictions,
            "augmented_predictions": augmented_predictions,
            "augmented_worker_packet": augmented_packet,
            "matrices": matrices,
        },
        "contains_gold": False,
        "did_reveal_gold": False,
        "did_run_context_worker": True,
        "did_run_acoustic_models": True,
        "did_run_augmented_worker": True,
        "did_score_against_gold": False,
        "did_mutate_profiles_or_references": False,
        "did_enable_default_integration": False,
        "did_run_historical_reprocessing": False,
    }
    execution = {**core, "content_sha256": _canonical_hash(core)}
    write_immutable_private_json(paths["execution"], execution)
    return {key: execution[key] for key in (
        "schema_version", "status", "authority_content_sha256",
        "context_worker_packet_sha256", "context_predictions_sha256",
        "augmented_worker_packet_sha256", "augmented_predictions_sha256",
        "matrix_count", "matrix_set_sha256", "matrix_content_sha256s",
        "speaker_count", "trial_count", "contains_gold", "did_reveal_gold",
        "did_run_context_worker", "did_run_acoustic_models", "did_run_augmented_worker",
        "did_score_against_gold", "did_mutate_profiles_or_references",
        "did_enable_default_integration", "did_run_historical_reprocessing", "content_sha256",
    )}


def replay_generation5_e2_execution(
    expected_content_sha256: str, *, runtime_root: Path = DEFAULT_RUNTIME_ROOT,
) -> dict[str, Any]:
    replay_generation5_e2(expected_content_sha256, runtime_root=runtime_root)
    paths = _paths(runtime_root, expected_content_sha256)
    require_private_file(paths["execution"], paths["root"])
    execution = _read_json(paths["execution"])
    private = execution.get("private_evidence")
    matrices = private.get("matrices") if isinstance(private, Mapping) else None
    context = private.get("context_predictions") if isinstance(private, Mapping) else None
    augmented = private.get("augmented_predictions") if isinstance(private, Mapping) else None
    packet = private.get("augmented_worker_packet") if isinstance(private, Mapping) else None
    if not all(isinstance(item, (dict, list)) for item in (matrices, context, augmented, packet)):
        raise Generation5E2Error("Frozen E2 execution children are missing.")
    core = {key: value for key, value in execution.items() if key != "content_sha256"}
    if (
        _canonical_hash(core) != execution.get("content_sha256")
        or execution.get("status") != "gold_blind_paired_predictions_frozen"
        or execution.get("matrix_count") != EXPECTED_MATRIX_COUNT
        or execution.get("speaker_count") != EXPECTED_SPEAKER_COUNT
        or execution.get("trial_count") != EXPECTED_TRIAL_COUNT
        or execution.get("contains_gold") is not False
        or execution.get("did_reveal_gold") is not False
        or context.get("content_sha256") != execution.get("context_predictions_sha256")
        or augmented.get("content_sha256") != execution.get("augmented_predictions_sha256")
        or packet.get("content_sha256") != execution.get("augmented_worker_packet_sha256")
        or [item.get("content_sha256") for item in matrices] != execution.get("matrix_content_sha256s")
    ):
        raise Generation5E2Error("Frozen E2 execution drifted or is incomplete.")
    for matrix in matrices:
        path = paths["matrices"] / f"{matrix['candidate_id']}--{matrix['method_id']}.json"
        require_private_file(path, paths["root"])
        if _read_json(path) != matrix:
            raise Generation5E2Error("A frozen E2 matrix drifted.")
    portable = {key: execution[key] for key in execution if key != "private_evidence"}
    return {**portable, "replay_schema_version": REPLAY_SCHEMA, "idempotent_replay": True}
