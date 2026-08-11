"""Run Plan 0071 D2 private acoustic/context predictions without capture."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import threading
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import acoustic_verification
import app_intelligence_ledger
import provenance_config
import speaker_identity_plan0064_p1 as plan0064_p1
import speaker_identity_plan0064_p2 as plan0064_p2
import speaker_identity_plan0064_p3 as plan0064_p3
import speaker_identity_plan0071_d0 as d0
import speaker_identity_plan0071_d2_cohort as cohort
import speaker_identity_preprocess
import transcript_api
from acoustic_audio_derivatives import (
    ensure_private_tree,
    read_private_object,
    sha256_file,
    write_immutable_private_json,
)
from speaker_evaluation_baseline import LocalSpeakerCaseRunner


ACOUSTIC_SCHEMA = "transcribe-audio.plan0071-d2-acoustic-evidence.v1"
CASE_SCHEMA = "transcribe-audio.plan0071-d2-context-case.v1"
RESOLUTION_SCHEMA = "transcribe-audio.plan0071-d2-resolution.v1"
MANIFEST_SCHEMA = "transcribe-audio.plan0071-d2-predictions.v1"
RECEIPT_SCHEMA = "transcribe-audio.plan0071-d2-predictions-receipt.v1"
DEFAULT_RUNTIME_ROOT = d0.DEFAULT_RUNTIME_ROOT
DEFAULT_SOURCE_STORE_ROOT = Path("~/.transcripts")
DEFAULT_SOURCE_STATE_ROOT = Path("~/.local/state/transcribe-audio")
COHORT_CONTENT_SHA256 = (
    "ea37ea3879f467ce6604df53da55c184088e3a6a9accc21abf49eeb154b8f6c2"
)
EXPECTED_RECORDINGS = 6
EXPECTED_SLOTS = 18
PRIMARY_PROVIDER = "codex-app-server"
MUTATION_EFFECT_COUNTS = {
    key: 0 for key in d0.EFFECT_COUNTS if key != "model_turns"
}


class Plan0071D2PredictionError(ValueError):
    """Raised when the D2 prediction authority or zero-effect contract drifts."""


def _hash(value: Any) -> str:
    return d0._hash(value)


def _content(value: Mapping[str, Any]) -> dict[str, Any]:
    return d0._content(value)


def _validate_content(value: Mapping[str, Any], label: str) -> None:
    try:
        d0._validate_content(value, label)
    except d0.Plan0071D0Error as exc:
        raise Plan0071D2PredictionError(str(exc)) from exc


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], check=False, capture_output=True, text=True
    )
    if result.returncode:
        raise Plan0071D2PredictionError(
            result.stderr.strip() or "Git authority read failed."
        )
    return result.stdout.strip()


def _source_authority(*, require_clean: bool) -> dict[str, Any]:
    module = Path(__file__).resolve()
    root = Path(_git("rev-parse", "--show-toplevel")).resolve()
    relative = module.relative_to(root).as_posix()
    commit = _git("log", "-1", "--format=%H", "--", relative)
    committed = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        check=False,
        capture_output=True,
    )
    upstream = _git("rev-parse", "@{upstream}")
    module_sha256 = hashlib.sha256(module.read_bytes()).hexdigest()
    value = {
        "module_name": relative,
        "module_commit": commit,
        "module_sha256": module_sha256,
        "module_blob_matches": (
            committed.returncode == 0
            and module_sha256 == hashlib.sha256(committed.stdout).hexdigest()
        ),
        "clean": not _git("status", "--porcelain=v1"),
        "upstream_ahead": int(_git("rev-list", "--count", f"{upstream}..HEAD")),
        "upstream_behind": int(_git("rev-list", "--count", f"HEAD..{upstream}")),
    }
    if value["module_blob_matches"] is not True or (
        require_clean
        and (
            value["clean"] is not True
            or value["upstream_ahead"]
            or value["upstream_behind"]
        )
    ):
        raise Plan0071D2PredictionError(
            "D2 prediction source is not committed, clean, and upstream-even."
        )
    return value


def _paths(runtime_root: Path) -> dict[str, Path]:
    root = runtime_root.expanduser().resolve()
    run = root / f"d2-predictions-{COHORT_CONTENT_SHA256[:24]}"
    return {
        "root": root,
        "run": run,
        "state": run / "state",
        "store": run / "source-store-snapshot",
        "cases": run / "context-cases",
        "acoustic": run / "private-acoustic-evidence.json",
        "resolution": run / "private-resolution.json",
        "manifest": run / "private-manifest.json",
        "receipt": run / "receipt.json",
    }


def _selected(cohort_manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    selected = [
        dict(item)
        for item in cohort_manifest.get("considered") or []
        if isinstance(item, Mapping)
        and item.get("disposition") == "selected_supplemental_development"
    ]
    if len(selected) != EXPECTED_RECORDINGS or sum(
        len(item.get("speaker_labels") or []) for item in selected
    ) != EXPECTED_SLOTS:
        raise Plan0071D2PredictionError("The D2 cohort denominator drifted.")
    return selected


def _bound_authorities(runtime_root: Path) -> dict[str, Any]:
    cohort_receipt = cohort.replay_cohort(runtime_root=runtime_root)
    cohort_manifest = read_private_object(Path(cohort_receipt["manifest_path"]))
    _validate_content(cohort_manifest, "Plan 0071 D2 cohort manifest")
    if cohort_manifest.get("content_sha256") != COHORT_CONTENT_SHA256:
        raise Plan0071D2PredictionError("The D2 cohort content authority drifted.")
    d0_receipt = d0.replay_activation(runtime_root=runtime_root)
    d0_manifest = read_private_object(Path(d0_receipt["manifest_path"]))
    _validate_content(d0_manifest, "Plan 0071 D0 manifest")
    bindings = d0_manifest.get("artifact_bindings") or {}
    p0_binding = bindings.get("plan0064_p0_manifest") or {}
    p1_binding = bindings.get("plan0064_p1_evidence") or {}
    p0_path = Path(str(p0_binding.get("path") or ""))
    p1_path = Path(str(p1_binding.get("path") or ""))
    if (
        not p0_path.is_file()
        or sha256_file(p0_path) != p0_binding.get("file_sha256")
        or not p1_path.is_file()
        or sha256_file(p1_path) != p1_binding.get("file_sha256")
    ):
        raise Plan0071D2PredictionError("A frozen Plan 0064 acoustic input drifted.")
    p0_manifest = read_private_object(p0_path)
    p1_evidence = read_private_object(p1_path)
    profiles = list((p0_manifest.get("profile_inventory") or {}).get("active_profiles") or [])
    threshold_authority = p1_evidence.get("threshold_authority") or {}
    thresholds = {
        str(item.get("candidate_id") or ""): float(item.get("threshold"))
        for item in threshold_authority.get("units") or []
        if isinstance(item, Mapping)
    }
    candidate_ids = list(
        (p0_manifest.get("profile_inventory") or {}).get("candidate_ids") or []
    )
    if len(profiles) != 21 or set(thresholds) != set(candidate_ids):
        raise Plan0071D2PredictionError("The frozen acoustic matrix is incomplete.")
    return {
        "cohort_receipt": cohort_receipt,
        "cohort_manifest": cohort_manifest,
        "d0_manifest": d0_manifest,
        "p0_binding": dict(p0_binding),
        "p1_binding": dict(p1_binding),
        "profiles": profiles,
        "thresholds": thresholds,
        "threshold_authority": dict(threshold_authority),
        "candidate_ids": candidate_ids,
    }


def _artifact_snapshot(selected: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in selected:
        for field in ("transcript_artifact", "source_media_artifact"):
            binding = item.get(field) or {}
            path = Path(str(binding.get("path") or ""))
            expected = str(binding.get("file_sha256") or "")
            actual = sha256_file(path)
            if actual != expected:
                raise Plan0071D2PredictionError(
                    f"A selected D2 {field.replace('_', ' ')} drifted."
                )
            result[f"{item['document_id']}:{field}"] = actual
    return result


def build_acoustic_evidence(
    selected: Sequence[Mapping[str, Any]],
    *,
    profiles: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, float],
    threshold_authority: Mapping[str, Any],
    candidate_ids: Sequence[str],
    score_fn: Callable[..., Mapping[str, Any]] = acoustic_verification.score_profile,
    adapter_factory: Callable[[], Mapping[str, Any]] = acoustic_verification.adapter_registry,
    decode_fn: Callable[[Path], Any] = plan0064_p1._decode,
) -> dict[str, Any]:
    """Score the frozen D2 denominator with the unchanged Plan 0064 policy."""

    adapters = {
        key: plan0064_p1._CachingAdapter(value)
        for key, value in dict(adapter_factory()).items()
    }
    if set(adapters) != set(candidate_ids):
        raise Plan0071D2PredictionError("The acoustic adapter registry drifted.")
    recordings = []
    for index, item in enumerate(selected, start=1):
        print(
            f"Scoring Plan 0071 D2 recording {index}/{len(selected)}...",
            flush=True,
        )
        transcript_path = Path(str(item["transcript_artifact"]["path"]))
        media_path = Path(str(item["source_media_artifact"]["path"]))
        transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
        decoded = decode_fn(media_path)
        slots = [
            plan0064_p1._score_slot(
                document_id=str(item["document_id"]),
                speaker=str(label),
                probe=plan0064_p1._slot_probe(transcript, str(label), decoded),
                profiles=profiles,
                thresholds=thresholds,
                adapters=adapters,
                score_fn=score_fn,
                profile_root=plan0064_p1.DEFAULT_PROFILE_ROOT.expanduser().absolute(),
                reference_root=plan0064_p1.DEFAULT_REFERENCE_ROOT.expanduser().absolute(),
            )
            for label in item.get("speaker_labels") or []
        ]
        recordings.append(
            {
                "document_id": item["document_id"],
                "recording_time": item["recording_time"],
                "original_recording_filename": item["original_recording_filename"],
                "transcript_sha256": item["transcript_sha256"],
                "source_media_sha256": item["source_media_sha256"],
                "speaker_slots": slots,
            }
        )
    all_slots = [slot for item in recordings for slot in item["speaker_slots"]]
    return _content(
        {
            "schema_version": ACOUSTIC_SCHEMA,
            "status": "complete_private_shadow_acoustic_evidence",
            "cohort_content_sha256": COHORT_CONTENT_SHA256,
            "threshold_authority": dict(threshold_authority),
            "recordings": recordings,
            "summary": {
                "recording_count": len(recordings),
                "speaker_slot_count": len(all_slots),
                "status_counts": dict(
                    sorted(Counter(item["status"] for item in all_slots).items())
                ),
                "reason_code_counts": dict(
                    sorted(Counter(item["reason_code"] for item in all_slots).items())
                ),
            },
            "contains_biometric_scores": True,
            "contains_embeddings_or_vectors": False,
            "contains_raw_audio": False,
            "contains_gold": False,
            "did_change_thresholds": False,
            "mutation_effect_counts": dict(MUTATION_EFFECT_COUNTS),
        }
    )


def _prepare_private_store(
    source: Path,
    destination: Path,
    selected: Sequence[Mapping[str, Any]],
) -> None:
    """Snapshot the DB and redirect selected artifacts to private copies."""

    marker = destination.parent / "private-store-prepared.json"
    if destination.exists():
        if not marker.is_file():
            raise Plan0071D2PredictionError(
                "A partial private transcript-store snapshot exists."
            )
        return
    ensure_private_tree(destination.parent.parent, destination.parent)
    with sqlite3.connect(source) as source_connection, sqlite3.connect(
        destination
    ) as destination_connection:
        source_connection.backup(destination_connection)
    destination.chmod(0o600)
    transcript_root = destination.parent / "private-artifacts"
    media_root = destination.parent / "private-blobs"
    ensure_private_tree(destination.parent, transcript_root)
    ensure_private_tree(destination.parent, media_root)
    redirects = []
    with sqlite3.connect(destination) as connection:
        for item in selected:
            document_id = str(item["document_id"])
            transcript_source = Path(str(item["transcript_artifact"]["path"]))
            media_source = Path(str(item["source_media_artifact"]["path"]))
            transcript_copy = transcript_root / f"{document_id}.transcript.json"
            media_copy = media_root / f"{item['source_media_sha256']}.m4a"
            shutil.copyfile(transcript_source, transcript_copy)
            shutil.copyfile(media_source, media_copy)
            transcript_copy.chmod(0o600)
            media_copy.chmod(0o600)
            changed_document = connection.execute(
                "UPDATE documents SET stored_path = ?, source_path = ? WHERE id = ?",
                (str(transcript_copy), str(transcript_copy), document_id),
            ).rowcount
            changed_blobs = connection.execute(
                "UPDATE blobs SET stored_path = ? WHERE sha256 = ?",
                (str(media_copy), str(item["source_media_sha256"])),
            ).rowcount
            if changed_document != 1 or changed_blobs < 1:
                raise Plan0071D2PredictionError(
                    "A selected artifact could not be redirected in the private store."
                )
            redirects.append(
                {
                    "document_id": document_id,
                    "transcript_copy": str(transcript_copy),
                    "media_copy": str(media_copy),
                    "source_transcript_sha256": item["transcript_sha256"],
                    "source_media_sha256": item["source_media_sha256"],
                }
            )
        connection.commit()
    write_immutable_private_json(
        marker,
        {
            "schema_version": "transcribe-audio.plan0071-d2-private-store.v1",
            "source_database": str(source),
            "source_database_file_sha256": sha256_file(source),
            "redirects": redirects,
            "live_source_write_count": 0,
        },
    )


def _copy_provenance_config(source_state: Path, destination_state: Path) -> Path:
    source = provenance_config.config_path(state_root=source_state)
    destination = destination_state / provenance_config.DEFAULT_CONFIG_PATH.name
    ensure_private_tree(destination_state.parent, destination_state)
    if source.is_file() and not destination.exists():
        payload = json.loads(source.read_text(encoding="utf-8"))
        write_immutable_private_json(destination, payload)
    return destination


def _assert_primary(prepared: Mapping[str, Any]) -> None:
    packet = prepared.get("prompt_packet") or {}
    route = packet.get("route") if isinstance(packet, Mapping) else {}
    if not isinstance(route, Mapping) or route.get("provider") != PRIMARY_PROVIDER:
        raise Plan0071D2PredictionError("A D2 phase did not use the frozen primary route.")


def execute_context_case(
    runner: LocalSpeakerCaseRunner,
    *,
    document_id: str,
    speaker_labels: Sequence[str],
) -> dict[str, Any]:
    """Execute two phases and validate locally; deliberately never capture."""

    attempts = {"clue_discovery": 0, "identity_evaluation": 0}
    repairs = {"clue_discovery": 0, "identity_evaluation": 0}
    run_references: dict[str, str] = {}
    try:
        discovery = runner._post(
            f"/api/conversations/{document_id}/speaker-preprocessing/prepare-discovery",
            {},
        )
        _assert_primary(discovery)
        attempts["clue_discovery"] += 1
        runner._execute_prepared(discovery)
        run_references["clue_discovery_run_id"] = str(discovery["run_id"])
        try:
            evaluation = runner._post(
                f"/api/conversations/{document_id}/speaker-preprocessing/prepare-evaluation",
                {"clue_discovery_run_id": discovery["run_id"]},
            )
        except ValueError:
            repairs["clue_discovery"] += 1
            attempts["clue_discovery"] += 1
            repair, corrected = runner._execute_reference_repair(
                document_id,
                phase="clue_discovery",
                original=discovery,
            )
            run_references["clue_discovery_repair_run_id"] = str(repair["run_id"])
            evaluation = runner._post(
                f"/api/conversations/{document_id}/speaker-preprocessing/prepare-evaluation",
                {
                    "clue_discovery_run_id": discovery["run_id"],
                    "discovery_readout": corrected,
                },
            )
        _assert_primary(evaluation)
        attempts["identity_evaluation"] += 1
        status = runner._execute_prepared(evaluation)
        run_references["identity_evaluation_run_id"] = str(evaluation["run_id"])
        readout = runner._captured_json(status)
        try:
            validated = speaker_identity_preprocess.validate_and_score_identity_evaluation(
                evaluation["packet"], readout
            )
        except ValueError:
            repairs["identity_evaluation"] += 1
            attempts["identity_evaluation"] += 1
            repair, corrected = runner._execute_reference_repair(
                document_id,
                phase="identity_evaluation",
                original=evaluation,
            )
            run_references["identity_evaluation_repair_run_id"] = str(repair["run_id"])
            validated = speaker_identity_preprocess.validate_and_score_identity_evaluation(
                evaluation["packet"], corrected
            )
        canonical_people = {
            str(item.get("person_id") or "")
            for item in evaluation["packet"].get("people") or []
            if isinstance(item, Mapping) and item.get("person_id")
        }
        case = plan0064_p2._successful_case(
            document_id=document_id,
            speaker_labels=speaker_labels,
            result={
                "prediction": validated["readout"],
                "run_references": run_references,
                "execution_provider": PRIMARY_PROVIDER,
            },
            canonical_people=canonical_people,
        )
        core = {key: value for key, value in case.items() if key != "content_sha256"}
        core.update(
            {
                "schema_version": CASE_SCHEMA,
                "phase_turn_attempts": attempts,
                "reference_repair_counts": repairs,
                "fallback_model_turn_count": 0,
                "capture_evaluation_call_count": 0,
                "contains_gold": False,
                "mutation_effect_counts": dict(MUTATION_EFFECT_COUNTS),
            }
        )
        return _content(core)
    except Exception as exc:
        failure = plan0064_p2._failure_case(
            document_id=document_id,
            speaker_labels=speaker_labels,
            stage="private_validation_failed",
            message=str(exc),
            run_references=run_references,
        )
        core = {key: value for key, value in failure.items() if key != "content_sha256"}
        core.update(
            {
                "schema_version": CASE_SCHEMA,
                "phase_turn_attempts": attempts,
                "reference_repair_counts": repairs,
                "fallback_model_turn_count": 0,
                "capture_evaluation_call_count": 0,
                "contains_gold": False,
                "mutation_effect_counts": dict(MUTATION_EFFECT_COUNTS),
            }
        )
        return _content(core)


def _context_cases(
    *,
    paths: Mapping[str, Path],
    selected: Sequence[Mapping[str, Any]],
    source_store_root: Path,
    source_state_root: Path,
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    source_database = source_store_root.expanduser().resolve() / "transcripts.sqlite3"
    snapshot_database = paths["store"] / "transcripts.sqlite3"
    _prepare_private_store(source_database, snapshot_database, selected)
    private_config = _copy_provenance_config(
        source_state_root.expanduser().resolve(), paths["state"]
    )
    server = transcript_api.TranscriptApiServer(
        ("127.0.0.1", 0),
        transcript_api.TranscriptApiHandler,
        store_root=paths["store"],
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
        state_root=paths["state"],
        quiet=True,
        static_dir=None,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    prior_config = os.environ.get(provenance_config.ENV_CONFIG_PATH)
    if private_config.is_file():
        os.environ[provenance_config.ENV_CONFIG_PATH] = str(private_config)
    runner = LocalSpeakerCaseRunner(
        base_url=f"http://127.0.0.1:{server.server_address[1]}",
        timeout_seconds=timeout_seconds,
    )
    thread.start()
    cases: list[dict[str, Any]] = []
    try:
        for index, item in enumerate(selected, start=1):
            document_id = str(item["document_id"])
            case_path = paths["cases"] / f"{document_id}.json"
            if case_path.exists():
                case = read_private_object(case_path)
                _validate_content(case, f"D2 context case {document_id}")
            else:
                print(
                    f"Running Plan 0071 D2 context case {index}/{len(selected)}...",
                    flush=True,
                )
                case = execute_context_case(
                    runner,
                    document_id=document_id,
                    speaker_labels=[str(value) for value in item["speaker_labels"]],
                )
                case["original_recording_filename"] = item[
                    "original_recording_filename"
                ]
                case = _content(
                    {key: value for key, value in case.items() if key != "content_sha256"}
                )
                write_immutable_private_json(case_path, case)
            cases.append(case)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        if prior_config is None:
            os.environ.pop(provenance_config.ENV_CONFIG_PATH, None)
        else:
            os.environ[provenance_config.ENV_CONFIG_PATH] = prior_config
    return cases


def _resolve(
    acoustic: Mapping[str, Any], cases: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    context_by_document = {str(item["document_id"]): item for item in cases}
    recordings = [
        plan0064_p3.resolve_conversation(
            recording, context_by_document[str(recording["document_id"])]
        )
        for recording in acoustic.get("recordings") or []
    ]
    slots = [slot for recording in recordings for slot in recording["speaker_slots"]]
    return _content(
        {
            "schema_version": RESOLUTION_SCHEMA,
            "status": "complete_private_shadow_resolution",
            "cohort_content_sha256": COHORT_CONTENT_SHA256,
            "acoustic_content_sha256": acoustic["content_sha256"],
            "context_case_content_sha256s": [item["content_sha256"] for item in cases],
            "recordings": recordings,
            "summary": {
                "recording_count": len(recordings),
                "speaker_slot_count": len(slots),
                "condition_disposition_counts": {
                    condition: dict(
                        sorted(
                            Counter(
                                slot[condition]["disposition"] for slot in slots
                            ).items()
                        )
                    )
                    for condition in (
                        "acoustic",
                        "context",
                        "combined",
                        "residual_policy",
                    )
                },
                "combined_reason_code_counts": dict(
                    sorted(Counter(slot["combined"]["reason_code"] for slot in slots).items())
                ),
                "residual_acceptance_count": sum(
                    slot["residual_policy"]["reason_code"]
                    == "two_known_plus_one_independently_supported_residual"
                    for slot in slots
                ),
            },
            "contains_gold": False,
            "mutation_effect_counts": dict(MUTATION_EFFECT_COUNTS),
        }
    )


def execute_predictions(
    *,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    source_store_root: Path = DEFAULT_SOURCE_STORE_ROOT,
    source_state_root: Path = DEFAULT_SOURCE_STATE_ROOT,
    timeout_seconds: float = 600.0,
) -> dict[str, Any]:
    """Execute or exactly replay the D2 private prediction packet."""

    paths = _paths(runtime_root)
    if paths["receipt"].exists():
        return replay_predictions(runtime_root=runtime_root)
    authority = _bound_authorities(runtime_root)
    selected = _selected(authority["cohort_manifest"])
    source_before = _artifact_snapshot(selected)
    source_authority = _source_authority(require_clean=True)
    ensure_private_tree(paths["root"], paths["cases"])
    if paths["acoustic"].exists():
        acoustic = read_private_object(paths["acoustic"])
        _validate_content(acoustic, "D2 acoustic evidence")
    else:
        acoustic = build_acoustic_evidence(
            selected,
            profiles=authority["profiles"],
            thresholds=authority["thresholds"],
            threshold_authority=authority["threshold_authority"],
            candidate_ids=authority["candidate_ids"],
        )
        write_immutable_private_json(paths["acoustic"], acoustic)
    cases = _context_cases(
        paths=paths,
        selected=selected,
        source_store_root=source_store_root,
        source_state_root=source_state_root,
        timeout_seconds=timeout_seconds,
    )
    source_after = _artifact_snapshot(selected)
    if source_after != source_before:
        raise Plan0071D2PredictionError("A live selected source artifact changed.")
    resolution = _resolve(acoustic, cases)
    write_immutable_private_json(paths["resolution"], resolution)
    phase_attempts = {
        phase: sum(int(item["phase_turn_attempts"][phase]) for item in cases)
        for phase in ("clue_discovery", "identity_evaluation")
    }
    repair_counts = {
        phase: sum(int(item["reference_repair_counts"][phase]) for item in cases)
        for phase in ("clue_discovery", "identity_evaluation")
    }
    context_status_counts = dict(
        sorted(Counter(str(item.get("status") or "") for item in cases).items())
    )
    manifest = _content(
        {
            "schema_version": MANIFEST_SCHEMA,
            "status": "d2_predictions_complete_zero_mutation_effect",
            "source_authority": source_authority,
            "cohort_content_sha256": COHORT_CONTENT_SHA256,
            "cohort_manifest_file_sha256": sha256_file(
                Path(authority["cohort_receipt"]["manifest_path"])
            ),
            "plan0064_p0_binding": authority["p0_binding"],
            "plan0064_p1_binding": authority["p1_binding"],
            "selected_document_ids": [item["document_id"] for item in selected],
            "selected_original_recording_filenames": [
                item["original_recording_filename"] for item in selected
            ],
            "source_artifact_snapshot_sha256": _hash(source_before),
            "private_source_store_snapshot_sha256": sha256_file(
                paths["store"] / "transcripts.sqlite3"
            ),
            "acoustic_content_sha256": acoustic["content_sha256"],
            "acoustic_file_sha256": sha256_file(paths["acoustic"]),
            "context_case_content_sha256s": [item["content_sha256"] for item in cases],
            "context_status_counts": context_status_counts,
            "resolution_content_sha256": resolution["content_sha256"],
            "resolution_file_sha256": sha256_file(paths["resolution"]),
            "execution_counts": {
                "primary_phase_turn_attempts": phase_attempts,
                "reference_repairs": repair_counts,
                "fallback_model_turns": 0,
                "capture_evaluation_calls": 0,
            },
            "recording_count": EXPECTED_RECORDINGS,
            "speaker_slot_count": EXPECTED_SLOTS,
            "original_recording_filename_count": EXPECTED_RECORDINGS,
            "human_gold_read": False,
            "fresh_evaluation_allowed": False,
            "mutation_effect_counts": dict(MUTATION_EFFECT_COUNTS),
        }
    )
    write_immutable_private_json(paths["manifest"], manifest)
    receipt = _content(
        {
            "schema_version": RECEIPT_SCHEMA,
            "status": manifest["status"],
            "manifest_content_sha256": manifest["content_sha256"],
            "manifest_file_sha256": sha256_file(paths["manifest"]),
            "resolution_content_sha256": resolution["content_sha256"],
            "summary": resolution["summary"],
            "execution_counts": manifest["execution_counts"],
            "human_gold_read": False,
            "fresh_evaluation_allowed": False,
            "mutation_effect_counts": dict(MUTATION_EFFECT_COUNTS),
        }
    )
    write_immutable_private_json(paths["receipt"], receipt)
    return {
        **receipt,
        "manifest_path": str(paths["manifest"]),
        "resolution_path": str(paths["resolution"]),
        "idempotent_replay": False,
    }


def replay_predictions(*, runtime_root: Path = DEFAULT_RUNTIME_ROOT) -> dict[str, Any]:
    paths = _paths(runtime_root)
    manifest = read_private_object(paths["manifest"])
    receipt = read_private_object(paths["receipt"])
    resolution = read_private_object(paths["resolution"])
    acoustic = read_private_object(paths["acoustic"])
    for label, value in (
        ("D2 predictions manifest", manifest),
        ("D2 predictions receipt", receipt),
        ("D2 resolution", resolution),
        ("D2 acoustic evidence", acoustic),
    ):
        _validate_content(value, label)
    current_source = _source_authority(require_clean=False)
    if (
        receipt.get("manifest_content_sha256") != manifest["content_sha256"]
        or receipt.get("manifest_file_sha256") != sha256_file(paths["manifest"])
        or manifest.get("resolution_content_sha256") != resolution["content_sha256"]
        or manifest.get("resolution_file_sha256") != sha256_file(paths["resolution"])
        or manifest.get("acoustic_content_sha256") != acoustic["content_sha256"]
        or manifest.get("acoustic_file_sha256") != sha256_file(paths["acoustic"])
        or manifest.get("source_authority", {}).get("module_sha256")
        != current_source.get("module_sha256")
        or manifest.get("human_gold_read") is not False
        or manifest.get("fresh_evaluation_allowed") is not False
        or manifest.get("mutation_effect_counts") != MUTATION_EFFECT_COUNTS
        or any(MUTATION_EFFECT_COUNTS.values())
    ):
        raise Plan0071D2PredictionError("The D2 prediction replay drifted.")
    return {
        **receipt,
        "manifest_path": str(paths["manifest"]),
        "resolution_path": str(paths["resolution"]),
        "idempotent_replay": True,
    }


if __name__ == "__main__":
    print(json.dumps(execute_predictions(), indent=2, sort_keys=True))
