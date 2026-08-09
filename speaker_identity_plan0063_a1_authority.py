"""Exact A1 request and literal authorization for Plan 0063 live apply.

This module prepares and freezes authority only.  It has no live apply entry
point and cannot mutate conversation-knowledge, biometric-reference, or model
profile state.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import acoustic_biometric_references as references
import acoustic_verification as verification
import conversation_knowledge_store
import speaker_identity_plan0063_biometric_rehearsal as biometric_rehearsal
import speaker_identity_plan0063_private_rehearsal as canonical_rehearsal
import transcript_store
from acoustic_audio_derivatives import (
    canonical_artifact_hash,
    ensure_private_tree,
    read_private_object,
    require_private_file,
    sha256_file,
    write_immutable_private_json,
)


A1_REQUEST_SCHEMA = "transcribe-audio.plan0063-a1-request.v1"
A1_SUBMISSION_SCHEMA = "transcribe-audio.plan0063-a1-submission.v1"
A1_AUTHORITY_SCHEMA = "transcribe-audio.plan0063-a1-authority.v1"
DEFAULT_RUNTIME_ROOT = canonical_rehearsal.DEFAULT_RUNTIME_ROOT
SHA256_RE = re.compile(r"[a-f0-9]{64}")
UTC_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
APPROVAL_DECISION = "authorize_exact_live_apply"
MODULE_PATHS = (
    Path(__file__).resolve(),
    Path(canonical_rehearsal.__file__).resolve(),
    Path(biometric_rehearsal.__file__).resolve(),
    Path(conversation_knowledge_store.__file__).resolve(),
    Path(references.__file__).resolve(),
    Path(verification.__file__).resolve(),
)
AUTHORIZED_ACTIONS = {
    "migrate_conversation_knowledge_schema": True,
    "create_canonical_people": True,
    "save_reviewed_slot_observations": True,
    "save_reviewed_voice_binding": True,
    "register_biometric_references": True,
    "materialize_biometric_profiles": True,
    "quiesce_transcript_services": True,
    "restore_transcript_services": True,
    "write_provider_records": False,
    "write_graphiti": False,
    "perform_external_write": False,
    "reprocess_history": False,
}


class Plan0063A1AuthorityError(ValueError):
    """Raised when Plan 0063 A1 authority is incomplete or has drifted."""


def _fail(message: str) -> None:
    raise Plan0063A1AuthorityError(message)


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _canonical_utc(value: Any, field: str) -> str:
    selected = str(value or "")
    if not UTC_RE.fullmatch(selected):
        _fail(f"{field} must be canonical UTC without fractional seconds.")
    return selected


def _content_hash(value: Mapping[str, Any], field: str) -> str:
    claimed = str(value.get("content_sha256") or "")
    core = {key: item for key, item in value.items() if key != "content_sha256"}
    if not SHA256_RE.fullmatch(claimed) or canonical_artifact_hash(core) != claimed:
        _fail(f"The {field} content hash is invalid.")
    return claimed


def _git(arguments: Sequence[str], *, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", *arguments],
        cwd=Path(__file__).resolve().parent,
        capture_output=True,
        text=not binary,
        check=False,
    )
    if result.returncode:
        _fail("Repository authority could not be read.")
    return result.stdout if binary else result.stdout.strip()


def _repository_authority() -> dict[str, Any]:
    if _git(["status", "--porcelain=v1", "--untracked-files=normal"]):
        _fail("Repository must be clean before preparing A1.")
    parity = str(
        _git(["rev-list", "--left-right", "--count", "HEAD...@{upstream}"])
    ).split()
    if parity != ["0", "0"]:
        _fail("Repository must be upstream-even before preparing A1.")
    commit = str(_git(["rev-parse", "HEAD"]))
    upstream = str(_git(["rev-parse", "@{upstream}"]))
    if not re.fullmatch(r"[a-f0-9]{40}", commit) or upstream != commit:
        _fail("Repository commit authority is invalid.")
    modules: dict[str, str] = {}
    for path in MODULE_PATHS:
        body = _git(["show", f"{commit}:{path.name}"], binary=True)
        if not isinstance(body, bytes):
            _fail("A committed A1 module could not be read.")
        committed_sha256 = hashlib.sha256(body).hexdigest()
        if committed_sha256 != sha256_file(path):
            _fail("A committed A1 module differs from the worktree.")
        modules[path.name] = committed_sha256
    return {
        "commit": commit,
        "upstream": upstream,
        "ahead": 0,
        "behind": 0,
        "clean": True,
        "modules": modules,
    }


def _paths(runtime_root: Path, transition_sha256: str) -> dict[str, Path]:
    if not SHA256_RE.fullmatch(str(transition_sha256)):
        _fail("The reviewed transition hash is invalid.")
    root = runtime_root.expanduser().absolute()
    run = root / f"a1-{transition_sha256[:20]}"
    return {
        "root": root,
        "run": run,
        "request": run / "private-request.json",
        "submission": run / "literal-submission.txt",
        "authority": run / "private-authority.json",
    }


def _write_private_text(path: Path, content: str) -> None:
    if path.exists():
        require_private_file(path, path.parent.parent)
        if path.read_text(encoding="utf-8") != content:
            _fail("The immutable A1 submission differs from the requested write.")
        return
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _live_state(
    *,
    live_store_root: Path,
    live_reference_root: Path,
    live_profile_root: Path,
) -> dict[str, Any]:
    knowledge = canonical_rehearsal._database_snapshot(
        transcript_store.db_path(live_store_root)
    )
    reference = biometric_rehearsal._store_snapshot(
        live_reference_root,
        database_name=biometric_rehearsal.REFERENCE_DATABASE_NAME,
    )
    profile = biometric_rehearsal._store_snapshot(
        live_profile_root,
        database_name=biometric_rehearsal.PROFILE_DATABASE_NAME,
        names=biometric_rehearsal.PROFILE_STATE_NAMES,
    )
    snapshots = {
        "knowledge": canonical_artifact_hash(knowledge),
        "references": canonical_artifact_hash(reference),
        "profiles": canonical_artifact_hash(profile),
    }
    return {
        "snapshot_sha256s": snapshots,
        "combined_snapshot_sha256": canonical_artifact_hash(snapshots),
        "knowledge_quick_check": knowledge.get("quick_check"),
    }


def _replayed_transition_and_rehearsal(
    *,
    transition_sha256: str,
    live_store_root: Path,
    live_reference_root: Path,
    live_profile_root: Path,
    runtime_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    receipt = biometric_rehearsal.replay_complete_private_rehearsal(
        transition_sha256=transition_sha256,
        live_store_root=live_store_root,
        live_reference_root=live_reference_root,
        live_profile_root=live_profile_root,
        runtime_root=runtime_root,
    )
    transition_path = canonical_rehearsal.rehearsal_paths(
        runtime_root, transition_sha256
    )["transition"]
    require_private_file(transition_path, runtime_root.expanduser().absolute())
    transition = read_private_object(transition_path)
    if canonical_rehearsal.validate_reviewed_transition(transition) != transition_sha256:
        _fail("The reviewed transition replay drifted before A1.")
    receipt_path = Path(str(receipt.get("receipt_path") or ""))
    require_private_file(receipt_path, runtime_root.expanduser().absolute())
    if (
        receipt.get("schema_version") != biometric_rehearsal.COMPLETE_RECEIPT_SCHEMA
        or receipt.get("status") != "complete_private_apply_and_rollback_proved"
        or receipt.get("transition_sha256") != transition_sha256
        or receipt.get("logical_transition_apply_count") != 1
        or receipt.get("logical_transition_rollback_count") != 1
        or receipt.get("test_mode") is not False
        or receipt.get("a1_request_ready") is not True
        or receipt.get("a1_authorized") is not False
        or receipt.get("live_mutation_count") != 0
    ):
        _fail("A production-mode complete private rehearsal is required for A1.")
    return transition, receipt, receipt_path


def _request_core(
    *,
    transition: Mapping[str, Any],
    rehearsal: Mapping[str, Any],
    rehearsal_path: Path,
    live_state: Mapping[str, Any],
    repository: Mapping[str, Any],
    requested_at: str,
) -> dict[str, Any]:
    metrics = dict(transition.get("metrics") or {})
    return {
        "schema_version": A1_REQUEST_SCHEMA,
        "status": "awaiting_literal_operator_authorization",
        "requested_at": _canonical_utc(requested_at, "A1 request time"),
        "transition_sha256": transition["content_sha256"],
        "review_content_sha256": transition["review_content_sha256"],
        "review_submission_sha256": transition["review_submission_sha256"],
        "rehearsal_receipt_content_sha256": rehearsal["content_sha256"],
        "rehearsal_receipt_file_sha256": sha256_file(rehearsal_path),
        "repository_authority": dict(repository),
        "expected_live_state": dict(live_state),
        "requested_actions": dict(AUTHORIZED_ACTIONS),
        "expected_apply_counts": {
            "canonical_people": int(metrics.get("canonical_person_count") or 0),
            "slot_bindings": int(metrics.get("slot_binding_count") or 0),
            "voice_bindings": int(metrics.get("active_voice_binding_count") or 0),
            "references": int(rehearsal.get("applied_reference_count") or 0),
            "profiles": int(rehearsal.get("applied_profile_count") or 0),
            "sources": int(rehearsal.get("applied_source_count") or 0),
        },
        "authorization_scope": "one_exact_plan0063_local_live_apply",
        "a1_authorized": False,
        "live_mutation_count": 0,
    }


def _validate_request(
    request: Mapping[str, Any],
    *,
    transition: Mapping[str, Any],
    rehearsal: Mapping[str, Any],
    rehearsal_path: Path,
    live_state: Mapping[str, Any],
    repository: Mapping[str, Any],
) -> str:
    requested_at = _canonical_utc(request.get("requested_at"), "A1 request time")
    expected = _request_core(
        transition=transition,
        rehearsal=rehearsal,
        rehearsal_path=rehearsal_path,
        live_state=live_state,
        repository=repository,
        requested_at=requested_at,
    )
    expected_with_hash = {
        **expected,
        "content_sha256": canonical_artifact_hash(expected),
    }
    if dict(request) != expected_with_hash:
        _fail("The A1 request does not match current exact authority.")
    return _content_hash(request, "A1 request")


def replay_a1_request(
    transition_sha256: str,
    *,
    live_store_root: Path,
    live_reference_root: Path,
    live_profile_root: Path,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
) -> dict[str, Any]:
    """Replay A1 request authority against current code and live state."""

    paths = _paths(runtime_root, transition_sha256)
    require_private_file(paths["request"], paths["root"])
    request = read_private_object(paths["request"])
    transition, rehearsal, rehearsal_path = _replayed_transition_and_rehearsal(
        transition_sha256=transition_sha256,
        live_store_root=live_store_root,
        live_reference_root=live_reference_root,
        live_profile_root=live_profile_root,
        runtime_root=runtime_root,
    )
    live_state = _live_state(
        live_store_root=live_store_root,
        live_reference_root=live_reference_root,
        live_profile_root=live_profile_root,
    )
    request_sha256 = _validate_request(
        request,
        transition=transition,
        rehearsal=rehearsal,
        rehearsal_path=rehearsal_path,
        live_state=live_state,
        repository=_repository_authority(),
    )
    return {
        **request,
        "request_sha256": request_sha256,
        "request_path": str(paths["request"]),
        "idempotent_replay": True,
    }


def build_a1_request(
    transition_sha256: str,
    *,
    live_store_root: Path,
    live_reference_root: Path,
    live_profile_root: Path,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
) -> dict[str, Any]:
    """Freeze one private A1 request after exact production rehearsal proof."""

    paths = _paths(runtime_root, transition_sha256)
    if paths["request"].exists():
        return replay_a1_request(
            transition_sha256,
            live_store_root=live_store_root,
            live_reference_root=live_reference_root,
            live_profile_root=live_profile_root,
            runtime_root=runtime_root,
        )
    if paths["run"].exists():
        _fail("A partial A1 authority directory already exists.")
    transition, rehearsal, rehearsal_path = _replayed_transition_and_rehearsal(
        transition_sha256=transition_sha256,
        live_store_root=live_store_root,
        live_reference_root=live_reference_root,
        live_profile_root=live_profile_root,
        runtime_root=runtime_root,
    )
    repository = _repository_authority()
    live_state = _live_state(
        live_store_root=live_store_root,
        live_reference_root=live_reference_root,
        live_profile_root=live_profile_root,
    )
    core = _request_core(
        transition=transition,
        rehearsal=rehearsal,
        rehearsal_path=rehearsal_path,
        live_state=live_state,
        repository=repository,
        requested_at=_utc_now(),
    )
    request = {**core, "content_sha256": canonical_artifact_hash(core)}
    ensure_private_tree(paths["root"], paths["run"])
    write_immutable_private_json(paths["request"], request)
    return {
        **request,
        "request_sha256": request["content_sha256"],
        "request_path": str(paths["request"]),
        "idempotent_replay": False,
    }


def render_a1_answer_block(request: Mapping[str, Any]) -> str:
    """Render the exact five-line block required for literal A1 authority."""

    payload = {
        key: value
        for key, value in request.items()
        if key not in {"request_sha256", "request_path", "idempotent_replay"}
    }
    request_sha256 = _content_hash(payload, "A1 request")
    transition_sha256 = str(payload.get("transition_sha256") or "")
    rehearsal_sha256 = str(
        payload.get("rehearsal_receipt_content_sha256") or ""
    )
    if not SHA256_RE.fullmatch(transition_sha256) or not SHA256_RE.fullmatch(
        rehearsal_sha256
    ):
        _fail("The A1 answer block hashes are invalid.")
    return "\n".join(
        (
            f"PLAN0063_A1_SCHEMA={A1_SUBMISSION_SCHEMA}",
            f"PLAN0063_A1_REQUEST_SHA256={request_sha256}",
            f"PLAN0063_A1_TRANSITION_SHA256={transition_sha256}",
            f"PLAN0063_A1_REHEARSAL_SHA256={rehearsal_sha256}",
            f"PLAN0063_A1_DECISION={APPROVAL_DECISION}",
        )
    )


def _parse_answer_block(value: str) -> dict[str, str]:
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    if len(lines) != 5:
        _fail("The A1 answer block must contain exactly five non-empty lines.")
    parsed: dict[str, str] = {}
    for line in lines:
        if "=" not in line:
            _fail("An A1 answer line is invalid.")
        key, selected = line.split("=", 1)
        if key in parsed or not key or not selected:
            _fail("An A1 answer field is duplicated or empty.")
        parsed[key] = selected
    expected_keys = {
        "PLAN0063_A1_SCHEMA",
        "PLAN0063_A1_REQUEST_SHA256",
        "PLAN0063_A1_TRANSITION_SHA256",
        "PLAN0063_A1_REHEARSAL_SHA256",
        "PLAN0063_A1_DECISION",
    }
    if set(parsed) != expected_keys:
        _fail("The A1 answer block fields are incomplete or unknown.")
    return parsed


def replay_a1_authorization(
    transition_sha256: str,
    *,
    expected_request_sha256: str,
    live_store_root: Path,
    live_reference_root: Path,
    live_profile_root: Path,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
) -> dict[str, Any]:
    """Replay literal A1 authority against its unchanged request baseline."""

    request = replay_a1_request(
        transition_sha256,
        live_store_root=live_store_root,
        live_reference_root=live_reference_root,
        live_profile_root=live_profile_root,
        runtime_root=runtime_root,
    )
    if request.get("request_sha256") != expected_request_sha256:
        _fail("The expected A1 request hash does not match replay.")
    paths = _paths(runtime_root, transition_sha256)
    require_private_file(paths["submission"], paths["root"])
    require_private_file(paths["authority"], paths["root"])
    submission = paths["submission"].read_text(encoding="utf-8")
    parsed = _parse_answer_block(submission)
    authority = read_private_object(paths["authority"])
    authorized_at = _canonical_utc(
        authority.get("authorized_at"), "A1 authorization time"
    )
    expected_core = {
        "schema_version": A1_AUTHORITY_SCHEMA,
        "status": "authorized_for_one_exact_live_apply",
        "authorized_at": authorized_at,
        "request_sha256": expected_request_sha256,
        "transition_sha256": transition_sha256,
        "rehearsal_receipt_content_sha256": request[
            "rehearsal_receipt_content_sha256"
        ],
        "literal_submission_sha256": hashlib.sha256(
            submission.encode("utf-8")
        ).hexdigest(),
        "authorized_actions": dict(AUTHORIZED_ACTIONS),
        "expected_apply_counts": dict(request["expected_apply_counts"]),
        "authorization_scope": "one_exact_plan0063_local_live_apply",
        "a1_authorized": True,
        "live_mutation_count": 0,
    }
    expected_authority = {
        **expected_core,
        "content_sha256": canonical_artifact_hash(expected_core),
    }
    if (
        parsed
        != {
            "PLAN0063_A1_SCHEMA": A1_SUBMISSION_SCHEMA,
            "PLAN0063_A1_REQUEST_SHA256": expected_request_sha256,
            "PLAN0063_A1_TRANSITION_SHA256": transition_sha256,
            "PLAN0063_A1_REHEARSAL_SHA256": request[
                "rehearsal_receipt_content_sha256"
            ],
            "PLAN0063_A1_DECISION": APPROVAL_DECISION,
        }
        or authority != expected_authority
    ):
        _fail("The literal A1 authorization replay drifted.")
    authority_sha256 = _content_hash(authority, "A1 authority")
    return {
        **authority,
        "authority_sha256": authority_sha256,
        "authority_path": str(paths["authority"]),
        "request_path": request["request_path"],
        "idempotent_replay": True,
    }


def freeze_a1_authorization(
    answer_block: str,
    *,
    expected_request_sha256: str,
    transition_sha256: str,
    live_store_root: Path,
    live_reference_root: Path,
    live_profile_root: Path,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    authorized_at: str | None = None,
) -> dict[str, Any]:
    """Freeze literal operator authority without applying any live mutation."""

    request = replay_a1_request(
        transition_sha256,
        live_store_root=live_store_root,
        live_reference_root=live_reference_root,
        live_profile_root=live_profile_root,
        runtime_root=runtime_root,
    )
    if request.get("request_sha256") != expected_request_sha256:
        _fail("The supplied A1 request hash is stale or unknown.")
    expected_block = render_a1_answer_block(request)
    normalized = "\n".join(
        line.strip() for line in answer_block.splitlines() if line.strip()
    )
    if normalized != expected_block:
        _fail("Literal A1 authorization must match the exact requested block.")
    paths = _paths(runtime_root, transition_sha256)
    if paths["authority"].exists():
        return replay_a1_authorization(
            transition_sha256,
            expected_request_sha256=expected_request_sha256,
            live_store_root=live_store_root,
            live_reference_root=live_reference_root,
            live_profile_root=live_profile_root,
            runtime_root=runtime_root,
        )
    timestamp = _canonical_utc(
        authorized_at if authorized_at is not None else _utc_now(),
        "A1 authorization time",
    )
    _write_private_text(paths["submission"], normalized)
    core = {
        "schema_version": A1_AUTHORITY_SCHEMA,
        "status": "authorized_for_one_exact_live_apply",
        "authorized_at": timestamp,
        "request_sha256": expected_request_sha256,
        "transition_sha256": transition_sha256,
        "rehearsal_receipt_content_sha256": request[
            "rehearsal_receipt_content_sha256"
        ],
        "literal_submission_sha256": hashlib.sha256(
            normalized.encode("utf-8")
        ).hexdigest(),
        "authorized_actions": dict(AUTHORIZED_ACTIONS),
        "expected_apply_counts": dict(request["expected_apply_counts"]),
        "authorization_scope": "one_exact_plan0063_local_live_apply",
        "a1_authorized": True,
        "live_mutation_count": 0,
    }
    authority = {**core, "content_sha256": canonical_artifact_hash(core)}
    write_immutable_private_json(paths["authority"], authority)
    return {
        **authority,
        "authority_sha256": authority["content_sha256"],
        "authority_path": str(paths["authority"]),
        "request_path": request["request_path"],
        "idempotent_replay": False,
    }


__all__ = [
    "A1_AUTHORITY_SCHEMA",
    "A1_REQUEST_SCHEMA",
    "A1_SUBMISSION_SCHEMA",
    "APPROVAL_DECISION",
    "AUTHORIZED_ACTIONS",
    "DEFAULT_RUNTIME_ROOT",
    "Plan0063A1AuthorityError",
    "build_a1_request",
    "freeze_a1_authorization",
    "render_a1_answer_block",
    "replay_a1_authorization",
    "replay_a1_request",
]
