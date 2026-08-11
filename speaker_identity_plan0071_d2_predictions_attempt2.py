"""Run Plan 0071's second and final D2 context-harness attempt."""

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
from typing import Any, Mapping, Sequence

import provenance_config
import speaker_identity_plan0071_d2_predictions as attempt1
import transcript_api
from acoustic_audio_derivatives import (
    ensure_private_tree,
    read_private_object,
    sha256_file,
    write_immutable_private_json,
)
from speaker_evaluation_baseline import LocalSpeakerCaseRunner


MANIFEST_SCHEMA = "transcribe-audio.plan0071-d2-predictions-attempt2.v1"
RECEIPT_SCHEMA = "transcribe-audio.plan0071-d2-predictions-attempt2-receipt.v1"
DEFAULT_RUNTIME_ROOT = attempt1.DEFAULT_RUNTIME_ROOT
DEFAULT_SOURCE_STORE_ROOT = attempt1.DEFAULT_SOURCE_STORE_ROOT
DEFAULT_SOURCE_STATE_ROOT = attempt1.DEFAULT_SOURCE_STATE_ROOT
COHORT_CONTENT_SHA256 = attempt1.COHORT_CONTENT_SHA256
PRIOR_RECEIPT_CONTENT_SHA256 = (
    "94458b21dceabab024f7deed59544d1d0c696bbbddb2b7d94dfa05b6a61ca217"
)
PRIOR_MANIFEST_CONTENT_SHA256 = (
    "020fe3d077cb545a6500ca2c9fca4e759a159d7911a3d2c58e7b9ae342304ed0"
)
MUTATION_EFFECT_COUNTS = dict(attempt1.MUTATION_EFFECT_COUNTS)


class Plan0071D2Attempt2Error(ValueError):
    """Raised when the final D2 attempt or its predecessor binding drifts."""


def _content(value: Mapping[str, Any]) -> dict[str, Any]:
    return attempt1._content(value)


def _validate_content(value: Mapping[str, Any], label: str) -> None:
    try:
        attempt1._validate_content(value, label)
    except attempt1.Plan0071D2PredictionError as exc:
        raise Plan0071D2Attempt2Error(str(exc)) from exc


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], check=False, capture_output=True, text=True
    )
    if result.returncode:
        raise Plan0071D2Attempt2Error(
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
        raise Plan0071D2Attempt2Error(
            "D2 attempt-2 source is not committed, clean, and upstream-even."
        )
    return value


def _paths(runtime_root: Path) -> dict[str, Path]:
    root = runtime_root.expanduser().resolve()
    run = root / f"d2-predictions-attempt2-{COHORT_CONTENT_SHA256[:24]}"
    return {
        "root": root,
        "run": run,
        "state": run / "state",
        "store": run / "source-store-snapshot",
        "cases": run / "context-cases",
        "resolution": run / "private-resolution.json",
        "manifest": run / "private-manifest.json",
        "receipt": run / "receipt.json",
    }


def _prior_attempt(runtime_root: Path) -> dict[str, Any]:
    prior = attempt1.replay_predictions(runtime_root=runtime_root)
    prior_paths = attempt1._paths(runtime_root)
    manifest = read_private_object(prior_paths["manifest"])
    acoustic = read_private_object(prior_paths["acoustic"])
    _validate_content(manifest, "D2 prediction attempt-1 manifest")
    _validate_content(acoustic, "D2 attempt-1 acoustic evidence")
    counts = manifest.get("execution_counts") or {}
    phase_attempts = counts.get("primary_phase_turn_attempts") or {}
    if (
        prior.get("content_sha256") != PRIOR_RECEIPT_CONTENT_SHA256
        or manifest.get("content_sha256") != PRIOR_MANIFEST_CONTENT_SHA256
        or any(int(value) for value in phase_attempts.values())
        or counts.get("capture_evaluation_calls") != 0
        or manifest.get("context_status_counts")
        != {"context_workflow_unavailable": 6}
        or acoustic.get("summary", {}).get("speaker_slot_count") != 18
    ):
        raise Plan0071D2Attempt2Error("The fail-safe D2 first attempt drifted.")
    return {
        "receipt": prior,
        "manifest": manifest,
        "acoustic": acoustic,
        "paths": prior_paths,
    }


def _prepare_private_store(
    source: Path,
    destination: Path,
    selected: Sequence[Mapping[str, Any]],
) -> None:
    """Redirect both source and stored paths under the private store roots."""

    marker = destination.parent / "private-store-prepared.json"
    if destination.exists():
        if not marker.is_file():
            raise Plan0071D2Attempt2Error("A partial private store exists.")
        return
    ensure_private_tree(destination.parent.parent, destination.parent)
    with sqlite3.connect(source) as source_connection, sqlite3.connect(
        destination
    ) as destination_connection:
        source_connection.backup(destination_connection)
    destination.chmod(0o600)
    transcript_source_root = destination.parent / "private-source" / "plan0071"
    transcript_root = destination.parent / "artifacts" / "plan0071"
    media_root = destination.parent / "blobs" / "plan0071"
    ensure_private_tree(destination.parent, transcript_source_root)
    ensure_private_tree(destination.parent, transcript_root)
    ensure_private_tree(destination.parent, media_root)
    redirects = []
    with sqlite3.connect(destination) as connection:
        for item in selected:
            document_id = str(item["document_id"])
            transcript_source = Path(str(item["transcript_artifact"]["path"]))
            media_source = Path(str(item["source_media_artifact"]["path"]))
            transcript_source_copy = (
                transcript_source_root / f"{document_id}.transcript.json"
            )
            transcript_copy = transcript_root / f"{document_id}.transcript.json"
            media_copy = media_root / f"{item['source_media_sha256']}.m4a"
            shutil.copyfile(transcript_source, transcript_source_copy)
            shutil.copyfile(transcript_source, transcript_copy)
            shutil.copyfile(media_source, media_copy)
            transcript_source_copy.chmod(0o600)
            transcript_copy.chmod(0o600)
            media_copy.chmod(0o600)
            changed_document = connection.execute(
                "UPDATE documents SET stored_path = ?, source_path = ? WHERE id = ?",
                (str(transcript_copy), str(transcript_source_copy), document_id),
            ).rowcount
            changed_blobs = connection.execute(
                "UPDATE blobs SET stored_path = ? WHERE sha256 = ?",
                (str(media_copy), str(item["source_media_sha256"])),
            ).rowcount
            if changed_document != 1 or changed_blobs < 1:
                raise Plan0071D2Attempt2Error(
                    "A selected artifact could not be privately redirected."
                )
            redirects.append(
                {
                    "document_id": document_id,
                    "private_source_transcript_path": str(transcript_source_copy),
                    "private_transcript_path": str(transcript_copy),
                    "private_media_path": str(media_copy),
                    "source_transcript_sha256": item["transcript_sha256"],
                    "source_media_sha256": item["source_media_sha256"],
                }
            )
        connection.commit()
    write_immutable_private_json(
        marker,
        {
            "schema_version": "transcribe-audio.plan0071-d2-private-store-attempt2.v1",
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
        write_immutable_private_json(
            destination, json.loads(source.read_text(encoding="utf-8"))
        )
    return destination


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
    cases: list[dict[str, Any]] = []
    thread.start()
    try:
        for index, item in enumerate(selected, start=1):
            document_id = str(item["document_id"])
            case_path = paths["cases"] / f"{document_id}.json"
            if case_path.exists():
                case = read_private_object(case_path)
                _validate_content(case, f"D2 attempt-2 case {document_id}")
            else:
                print(
                    f"Running Plan 0071 D2 context attempt 2 case {index}/6...",
                    flush=True,
                )
                case = attempt1.execute_context_case(
                    runner,
                    document_id=document_id,
                    speaker_labels=[str(value) for value in item["speaker_labels"]],
                )
                case = _content(
                    {
                        **{
                            key: value
                            for key, value in case.items()
                            if key != "content_sha256"
                        },
                        "packet_attempt": 2,
                        "original_recording_filename": item[
                            "original_recording_filename"
                        ],
                    }
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


def execute_attempt2(
    *,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    source_store_root: Path = DEFAULT_SOURCE_STORE_ROOT,
    source_state_root: Path = DEFAULT_SOURCE_STATE_ROOT,
    timeout_seconds: float = 600.0,
) -> dict[str, Any]:
    paths = _paths(runtime_root)
    if paths["receipt"].exists():
        return replay_attempt2(runtime_root=runtime_root)
    prior = _prior_attempt(runtime_root)
    authority = attempt1._bound_authorities(runtime_root)
    selected = attempt1._selected(authority["cohort_manifest"])
    source_before = attempt1._artifact_snapshot(selected)
    source_authority = _source_authority(require_clean=True)
    ensure_private_tree(paths["root"], paths["cases"])
    cases = _context_cases(
        paths=paths,
        selected=selected,
        source_store_root=source_store_root,
        source_state_root=source_state_root,
        timeout_seconds=timeout_seconds,
    )
    if attempt1._artifact_snapshot(selected) != source_before:
        raise Plan0071D2Attempt2Error("A live selected source artifact changed.")
    resolution = attempt1._resolve(prior["acoustic"], cases)
    write_immutable_private_json(paths["resolution"], resolution)
    phase_attempts = {
        phase: sum(int(item["phase_turn_attempts"][phase]) for item in cases)
        for phase in ("clue_discovery", "identity_evaluation")
    }
    repairs = {
        phase: sum(int(item["reference_repair_counts"][phase]) for item in cases)
        for phase in ("clue_discovery", "identity_evaluation")
    }
    context_status_counts = dict(
        sorted(Counter(str(item.get("status") or "") for item in cases).items())
    )
    manifest = _content(
        {
            "schema_version": MANIFEST_SCHEMA,
            "status": "d2_predictions_attempt2_complete_zero_mutation_effect",
            "packet_attempt": 2,
            "source_authority": source_authority,
            "cohort_content_sha256": COHORT_CONTENT_SHA256,
            "prior_attempt_receipt_content_sha256": PRIOR_RECEIPT_CONTENT_SHA256,
            "prior_attempt_manifest_content_sha256": PRIOR_MANIFEST_CONTENT_SHA256,
            "prior_attempt_acoustic_content_sha256": prior["acoustic"][
                "content_sha256"
            ],
            "prior_attempt_acoustic_file_sha256": sha256_file(
                prior["paths"]["acoustic"]
            ),
            "selected_document_ids": [item["document_id"] for item in selected],
            "selected_original_recording_filenames": [
                item["original_recording_filename"] for item in selected
            ],
            "source_artifact_snapshot_sha256": attempt1._hash(source_before),
            "private_source_store_snapshot_sha256": sha256_file(
                paths["store"] / "transcripts.sqlite3"
            ),
            "context_case_content_sha256s": [item["content_sha256"] for item in cases],
            "context_status_counts": context_status_counts,
            "resolution_content_sha256": resolution["content_sha256"],
            "resolution_file_sha256": sha256_file(paths["resolution"]),
            "execution_counts": {
                "primary_phase_turn_attempts": phase_attempts,
                "reference_repairs": repairs,
                "fallback_model_turns": 0,
                "capture_evaluation_calls": 0,
            },
            "recording_count": 6,
            "speaker_slot_count": 18,
            "original_recording_filename_count": 6,
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
            "packet_attempt": 2,
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


def replay_attempt2(*, runtime_root: Path = DEFAULT_RUNTIME_ROOT) -> dict[str, Any]:
    paths = _paths(runtime_root)
    manifest = read_private_object(paths["manifest"])
    receipt = read_private_object(paths["receipt"])
    resolution = read_private_object(paths["resolution"])
    for label, value in (
        ("D2 attempt-2 manifest", manifest),
        ("D2 attempt-2 receipt", receipt),
        ("D2 attempt-2 resolution", resolution),
    ):
        _validate_content(value, label)
    current_source = _source_authority(require_clean=False)
    if (
        receipt.get("manifest_content_sha256") != manifest["content_sha256"]
        or receipt.get("manifest_file_sha256") != sha256_file(paths["manifest"])
        or manifest.get("resolution_content_sha256") != resolution["content_sha256"]
        or manifest.get("resolution_file_sha256") != sha256_file(paths["resolution"])
        or manifest.get("source_authority", {}).get("module_sha256")
        != current_source.get("module_sha256")
        or manifest.get("prior_attempt_receipt_content_sha256")
        != PRIOR_RECEIPT_CONTENT_SHA256
        or manifest.get("mutation_effect_counts") != MUTATION_EFFECT_COUNTS
        or manifest.get("human_gold_read") is not False
    ):
        raise Plan0071D2Attempt2Error("The D2 attempt-2 replay drifted.")
    return {
        **receipt,
        "manifest_path": str(paths["manifest"]),
        "resolution_path": str(paths["resolution"]),
        "idempotent_replay": True,
    }


if __name__ == "__main__":
    print(json.dumps(execute_attempt2(), indent=2, sort_keys=True))
