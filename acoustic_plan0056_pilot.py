from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Callable
from typing import Any

from acoustic_generation5_recovery_authority import _evidence_hashes
import acoustic_generation3_recalibration as generation3
from acoustic_audio_derivatives import (
    ensure_private_tree,
    read_private_object,
    require_private_file,
    write_immutable_private_json,
)


PROPOSAL_SCHEMA = "transcribe-audio.plan0056-pilot-proposals.v1"
AUTHORITY_SCHEMA = "transcribe-audio.plan0056-pilot-authority.v1"
AUTHORITY_MANIFEST_SCHEMA = "transcribe-audio.plan0056-pilot-authority-manifest.v1"
AUTHORITY_RECEIPT_SCHEMA = "transcribe-audio.plan0056-pilot-authority-receipt.v1"
AUTHORITY_REPLAY_SCHEMA = "transcribe-audio.plan0056-pilot-authority-replay.v1"
THRESHOLD_APPLICATION_SHA256 = (
    "308f326d3fe9baa175ed32c90df4255a8d4bfc1924c6f925eab490ae2832f4d1"
)
PLAN_PATH = Path(
    "docs/dev/plans/0056-2026-08-05-enrolled-only-acoustic-pilot-identity-guard.md"
)
MODULE_PATH = Path(__file__).name
DEFAULT_PRIOR_ROOT = Path("~/.local/state/transcribe-audio")
DEFAULT_RUNTIME_ROOT = Path("~/.local/state/transcribe-audio/plan-0056/p0")
DEFAULT_PRIMARY_STORE = Path("~/.transcripts/transcripts.sqlite3")
DEFAULT_KNOWLEDGE_STORE = Path(
    "~/.local/state/transcribe-audio/conversation-identity-shadow/transcripts.sqlite3"
)
DEFAULT_PROFILE_STORE = Path(
    "~/.local/state/transcribe-audio/plan-0037/verification-calibration/profiles.sqlite3"
)
DEFAULT_REFERENCE_STORE = Path(
    "~/.local/state/transcribe-audio/plan-0037/biometric-references/references.sqlite3"
)
SHA256_RE = re.compile(r"[a-f0-9]{64}")
_PROPOSAL_KEYS = {
    "speaker_ref",
    "disposition",
    "subject_id",
    "confidence_band",
    "rationale",
}


class Plan0056PilotError(ValueError):
    """Raised when the enrolled-only pilot cannot remain fail closed."""


def _canonical_hash(value: Any) -> str:
    body = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(arguments: Sequence[str], *, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", *arguments],
        cwd=Path(__file__).resolve().parent,
        capture_output=True,
        text=not binary,
        check=False,
    )
    if result.returncode:
        raise Plan0056PilotError("Repository authority is unavailable.")
    return result.stdout if binary else result.stdout.strip()


def _repository_authority() -> dict[str, Any]:
    if _git(["status", "--porcelain=v1", "--untracked-files=normal"]):
        raise Plan0056PilotError("Repository must be clean.")
    divergence = str(
        _git(["rev-list", "--left-right", "--count", "@{upstream}...HEAD"])
    ).split()
    if divergence != ["0", "0"]:
        raise Plan0056PilotError("Repository must be upstream-even.")
    commit = str(_git(["rev-parse", "HEAD"]))
    hashes: dict[str, str] = {}
    for relative in (MODULE_PATH, PLAN_PATH.as_posix()):
        committed = _git(["show", f"{commit}:{relative}"], binary=True)
        if not isinstance(committed, bytes):
            raise Plan0056PilotError("Committed Plan 0056 authority is unavailable.")
        current = Path(__file__).resolve().parent / relative
        digest = hashlib.sha256(committed).hexdigest()
        if digest != _sha256_file(current):
            raise Plan0056PilotError("Committed Plan 0056 authority drifted.")
        hashes[relative] = digest
    return {
        "commit": commit,
        "authority_file_sha256": hashes,
        "clean": True,
        "upstream_ahead": 0,
        "upstream_behind": 0,
    }


def _probe_media(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_name,sample_rate,channels",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise Plan0056PilotError("Pilot source media probe failed.")
    try:
        value = json.loads(result.stdout)
        streams = value["streams"]
        media_format = value["format"]
        stream = next(item for item in streams if item.get("sample_rate"))
        return {
            "duration_seconds": float(media_format["duration"]),
            "codec_name": str(stream["codec_name"]),
            "sample_rate": int(stream["sample_rate"]),
            "channels": int(stream["channels"]),
        }
    except (KeyError, TypeError, ValueError, StopIteration, json.JSONDecodeError) as exc:
        raise Plan0056PilotError("Pilot source media probe is invalid.") from exc


def _read_only_connection(path: Path) -> sqlite3.Connection:
    database = path.expanduser().absolute()
    if not database.is_file() or database.is_symlink():
        raise Plan0056PilotError("Identity state database is invalid.")
    try:
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        raise Plan0056PilotError("Identity state database is unavailable.") from exc
    connection.row_factory = sqlite3.Row
    return connection


def _table_counts(path: Path, tables: Sequence[str]) -> dict[str, int]:
    try:
        with _read_only_connection(path) as connection:
            return {
                table: int(connection.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0])
                for table in tables
            }
    except sqlite3.Error as exc:
        raise Plan0056PilotError("Identity state schema is unavailable.") from exc


def snapshot_identity_state(
    *,
    primary_store: Path,
    knowledge_store: Path,
    profile_store: Path,
    reference_store: Path,
) -> dict[str, Any]:
    """Read exact identity/profile cardinalities without opening a write transaction."""

    primary = _table_counts(
        primary_store,
        ("contacts", "speaker_assignments", "speaker_assignment_audits"),
    )
    knowledge = _table_counts(
        knowledge_store,
        (
            "knowledge_people",
            "knowledge_external_identities",
            "knowledge_relationships",
            "knowledge_current_person_profiles",
            "knowledge_review_decisions",
        ),
    )
    try:
        with _read_only_connection(profile_store) as connection:
            row = connection.execute(
                "SELECT count(*) AS total_profiles, "
                "sum(CASE WHEN lifecycle_state = 'active' THEN 1 ELSE 0 END) AS active_profiles, "
                "count(DISTINCT person_ref_id) AS distinct_subjects, "
                "count(DISTINCT generation_id) AS distinct_generations FROM profiles"
            ).fetchone()
            generation_rows = [
                dict(item)
                for item in connection.execute(
                    "SELECT profile_id, person_ref_id, generation_id, lifecycle_state "
                    "FROM profiles ORDER BY profile_id"
                ).fetchall()
            ]
    except sqlite3.Error as exc:
        raise Plan0056PilotError("Acoustic profile state is unavailable.") from exc
    acoustic_profiles = {
        "total_profiles": int(row["total_profiles"]),
        "active_profiles": int(row["active_profiles"] or 0),
        "distinct_subjects": int(row["distinct_subjects"]),
        "distinct_generations": int(row["distinct_generations"]),
        "generation_rows_sha256": _canonical_hash(generation_rows),
    }
    references = _table_counts(
        reference_store,
        ("profiles", "generations", "person_heads", "source_claims", "descendants"),
    )
    core = {
        "schema_version": "transcribe-audio.plan0056-identity-state-snapshot.v1",
        "primary": primary,
        "knowledge": knowledge,
        "acoustic_profiles": acoustic_profiles,
        "references": references,
        "database_paths": {
            "primary": str(primary_store.expanduser().absolute()),
            "knowledge": str(knowledge_store.expanduser().absolute()),
            "profiles": str(profile_store.expanduser().absolute()),
            "references": str(reference_store.expanduser().absolute()),
        },
    }
    return {**core, "snapshot_sha256": _canonical_hash(core)}


def _authority_paths(runtime_root: Path, content_sha256: str) -> dict[str, Path]:
    root = runtime_root.expanduser().absolute()
    run = root / f"pilot-authority-{content_sha256[:24]}"
    return {
        "root": root,
        "run": run,
        "manifest": run / "private-manifest.json",
        "receipt": run / "receipt.json",
    }


def _authority_receipt(preview: Mapping[str, Any], manifest_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": AUTHORITY_RECEIPT_SCHEMA,
        "status": "frozen_pre_model_authority",
        "preview_content_sha256": preview["content_sha256"],
        "manifest_sha256": manifest_sha256,
        "source_count": preview["source_count"],
        "source_set_sha256": preview["source_set_sha256"],
        "allowlisted_subject_ids": preview["allowlisted_subject_ids"],
        "profile_set_sha256": preview["profile_summary"]["profile_set_sha256"],
        "identity_state_before_sha256": preview["identity_state_before"]["snapshot_sha256"],
        "scoring_policy": preview["scoring_policy"],
        "action_vector": preview["action_vector"],
        "mode": "0600",
        "did_decode_audio": False,
        "did_run_models_or_predictions": False,
        "did_read_pilot_outcome_gold": False,
        "did_mutate_identity_or_profile_state": False,
    }


def freeze_plan0056_authority(
    reviewed_preview: Mapping[str, Any],
    *,
    expected_content_sha256: str,
    runtime_root: Path,
) -> dict[str, Any]:
    """Persist the reviewed pre-model authority in a private immutable tree."""

    preview = dict(reviewed_preview)
    core = {key: value for key, value in preview.items() if key != "content_sha256"}
    if (
        preview.get("schema_version") != AUTHORITY_SCHEMA
        or preview.get("content_sha256") != expected_content_sha256
        or _canonical_hash(core) != expected_content_sha256
        or any(preview.get("action_vector", {}).values())
    ):
        raise Plan0056PilotError("Reviewed Plan 0056 authority is stale or unsafe.")
    paths = _authority_paths(runtime_root, expected_content_sha256)
    if paths["receipt"].exists():
        return replay_plan0056_authority(
            expected_content_sha256, runtime_root=runtime_root
        )
    ensure_private_tree(paths["root"], paths["run"])
    manifest = {
        "schema_version": AUTHORITY_MANIFEST_SCHEMA,
        "status": "frozen_pre_model_authority",
        "preview": preview,
    }
    write_immutable_private_json(paths["manifest"], manifest)
    receipt = _authority_receipt(preview, _sha256_file(paths["manifest"]))
    write_immutable_private_json(paths["receipt"], receipt)
    return {**receipt, "idempotent_replay": False}


def replay_plan0056_authority(
    expected_content_sha256: str,
    *,
    runtime_root: Path,
) -> dict[str, Any]:
    """Replay the frozen authority and recheck every source binding."""

    paths = _authority_paths(runtime_root, expected_content_sha256)
    try:
        require_private_file(paths["manifest"], paths["root"])
        require_private_file(paths["receipt"], paths["root"])
        manifest = read_private_object(paths["manifest"])
        receipt = read_private_object(paths["receipt"])
    except (OSError, ValueError) as exc:
        raise Plan0056PilotError("Frozen Plan 0056 authority is unavailable.") from exc
    preview = manifest.get("preview")
    if not isinstance(preview, Mapping):
        raise Plan0056PilotError("Frozen Plan 0056 authority is invalid.")
    core = {key: value for key, value in preview.items() if key != "content_sha256"}
    expected_manifest = {
        "schema_version": AUTHORITY_MANIFEST_SCHEMA,
        "status": "frozen_pre_model_authority",
        "preview": dict(preview),
    }
    expected_receipt = _authority_receipt(preview, _sha256_file(paths["manifest"]))
    if (
        manifest != expected_manifest
        or receipt != expected_receipt
        or preview.get("content_sha256") != expected_content_sha256
        or _canonical_hash(core) != expected_content_sha256
    ):
        raise Plan0056PilotError("Frozen Plan 0056 authority drifted.")
    for source in preview.get("private_evidence", {}).get("sources", []):
        path = Path(str(source.get("path") or ""))
        if not path.is_file() or path.is_symlink() or _sha256_file(path) != source.get("source_sha256"):
            raise Plan0056PilotError("Frozen pilot source drifted.")
    return {
        **receipt,
        "replay_schema_version": AUTHORITY_REPLAY_SCHEMA,
        "idempotent_replay": True,
    }


def portable_authority(preview: Mapping[str, Any]) -> dict[str, Any]:
    """Return the reviewable authority without source paths or profile rows."""

    profile_summary = preview.get("profile_summary", {})
    return {
        "schema_version": preview.get("schema_version"),
        "status": preview.get("status"),
        "content_sha256": preview.get("content_sha256"),
        "repository_authority": preview.get("repository_authority"),
        "allowlisted_subject_ids": preview.get("allowlisted_subject_ids"),
        "profile_summary": {
            key: profile_summary.get(key)
            for key in (
                "profile_count",
                "subject_count",
                "candidate_count",
                "profile_set_sha256",
                "model_asset_set_sha256",
            )
        },
        "identity_state_before_sha256": preview.get("identity_state_before", {}).get(
            "snapshot_sha256"
        ),
        "source_count": preview.get("source_count"),
        "source_set_sha256": preview.get("source_set_sha256"),
        "prior_exclusion": preview.get("prior_exclusion"),
        "scoring_policy": preview.get("scoring_policy"),
        "action_vector": preview.get("action_vector"),
        "contains_pilot_outcome_gold": preview.get("contains_pilot_outcome_gold"),
        "contains_private_paths": False,
        "contains_profile_rows": False,
    }


def build_live_plan0056_authority(
    source_paths: Sequence[Path],
    *,
    prior_root: Path = DEFAULT_PRIOR_ROOT,
) -> dict[str, Any]:
    """Resolve current read-only state and build the live pre-model authority."""

    profiles = generation3._active_profiles(
        calibration_root=generation3.DEFAULT_CALIBRATION_ROOT,
        p3_runtime_root=generation3.DEFAULT_P3_RUNTIME_ROOT,
    )
    identity_state = snapshot_identity_state(
        primary_store=DEFAULT_PRIMARY_STORE,
        knowledge_store=DEFAULT_KNOWLEDGE_STORE,
        profile_store=DEFAULT_PROFILE_STORE,
        reference_store=DEFAULT_REFERENCE_STORE,
    )
    return preview_plan0056_authority(
        source_paths=source_paths,
        prior_root=prior_root,
        profile_inventory=profiles,
        identity_state_snapshot=identity_state,
        repository_authority=_repository_authority(),
        probe=_probe_media,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze and replay the Plan 0056 pre-model pilot authority."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("preview", "freeze"):
        child = subparsers.add_parser(command)
        child.add_argument("--source", action="append", required=True, type=Path)
        child.add_argument("--prior-root", type=Path, default=DEFAULT_PRIOR_ROOT)
        if command == "freeze":
            child.add_argument("--expected-content-sha256", required=True)
            child.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    replay = subparsers.add_parser("replay")
    replay.add_argument("--content-sha256", required=True)
    replay.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "replay":
        if not SHA256_RE.fullmatch(args.content_sha256):
            raise Plan0056PilotError("Authority content hash is invalid.")
        result = replay_plan0056_authority(
            args.content_sha256, runtime_root=args.runtime_root
        )
    else:
        preview = build_live_plan0056_authority(
            args.source, prior_root=args.prior_root
        )
        if args.command == "preview":
            result = portable_authority(preview)
        else:
            if args.expected_content_sha256 != preview["content_sha256"]:
                raise Plan0056PilotError("Reviewed authority hash is stale.")
            result = freeze_plan0056_authority(
                preview,
                expected_content_sha256=args.expected_content_sha256,
                runtime_root=args.runtime_root,
            )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def preview_plan0056_authority(
    *,
    source_paths: Sequence[Path],
    prior_root: Path,
    profile_inventory: tuple[list[dict[str, Any]], dict[str, Any]],
    identity_state_snapshot: Mapping[str, Any],
    repository_authority: Mapping[str, Any],
    probe: Callable[[Path], Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the pre-model authority for an exact, prior-disjoint pilot set."""

    if not 1 <= len(source_paths) <= 3:
        raise Plan0056PilotError("The pilot requires between one and three sources.")
    if (
        repository_authority.get("clean") is not True
        or repository_authority.get("upstream_ahead") != 0
        or repository_authority.get("upstream_behind") != 0
    ):
        raise Plan0056PilotError("Repository authority must be clean and upstream-even.")

    root = prior_root.expanduser().absolute()
    if not root.is_dir() or root.is_symlink():
        raise Plan0056PilotError("Prior evidence root is invalid.")
    prior_hashes: set[str] = set()
    evidence_file_hashes: list[str] = []
    for path in sorted(root.rglob("*.json")):
        if not path.is_file() or path.is_symlink():
            continue
        found, _parse_mode = _evidence_hashes(path)
        prior_hashes.update(found)
        evidence_file_hashes.append(_sha256_file(path))
    if not evidence_file_hashes:
        raise Plan0056PilotError("Prior evidence is empty.")

    sources: list[dict[str, Any]] = []
    for ordinal, supplied in enumerate(source_paths, start=1):
        path = supplied.expanduser().absolute()
        if not path.is_file() or path.is_symlink():
            raise Plan0056PilotError("Pilot source is invalid.")
        digest = _sha256_file(path)
        if digest in prior_hashes:
            raise Plan0056PilotError("Pilot source has prior evidence overlap.")
        media_probe = dict(probe(path))
        if float(media_probe.get("duration_seconds") or 0.0) < 60.0:
            raise Plan0056PilotError("Pilot source duration is below 60 seconds.")
        sources.append(
            {
                "ordinal": ordinal,
                "path": str(path),
                "source_sha256": digest,
                "probe": media_probe,
            }
        )

    profiles, profile_summary = profile_inventory
    subject_ids = sorted({str(item.get("person_ref_id") or "") for item in profiles})
    candidate_ids = {str(item.get("candidate_id") or "") for item in profiles}
    if (
        len(profiles) != 6
        or len(subject_ids) != 2
        or len(candidate_ids) != 3
        or any(
            sum(item.get("candidate_id") == candidate for item in profiles) != 2
            for candidate in candidate_ids
        )
        or profile_summary.get("profile_count") != 6
        or profile_summary.get("subject_count") != 2
        or profile_summary.get("candidate_count") != 3
    ):
        raise Plan0056PilotError("Exactly six profiles for two subjects are required.")
    if not str(identity_state_snapshot.get("snapshot_sha256") or ""):
        raise Plan0056PilotError("Identity state snapshot is invalid.")

    scoring_policy = {
        "schema_version": "transcribe-audio.plan0056-consensus-policy.v1",
        "threshold_application_sha256": THRESHOLD_APPLICATION_SHA256,
        "acoustic_unit_count": 9,
        "unit_support_rule": (
            "winner_score_at_or_above_unit_threshold_and_other_score_below_threshold"
        ),
        "assignment_minimum_supporting_units": 6,
        "assignment_minimum_candidate_families": 2,
        "assignment_maximum_opposing_units": 0,
        "high_confidence_supporting_units": 9,
        "review_when_any_threshold_support_without_assignment": True,
        "abstain_when_no_threshold_support": True,
        "human_confirmation_required": True,
    }
    core = {
        "schema_version": AUTHORITY_SCHEMA,
        "status": "ready_to_freeze",
        "repository_authority": dict(repository_authority),
        "allowlisted_subject_ids": subject_ids,
        "profile_summary": dict(profile_summary),
        "scoring_policy": scoring_policy,
        "identity_state_before": dict(identity_state_snapshot),
        "source_count": len(sources),
        "source_set_sha256": _canonical_hash(
            [item["source_sha256"] for item in sources]
        ),
        "prior_exclusion": {
            "json_file_count": len(evidence_file_hashes),
            "json_file_set_sha256": _canonical_hash(evidence_file_hashes),
            "excluded_hash_count": len(prior_hashes),
            "excluded_hash_set_sha256": _canonical_hash(sorted(prior_hashes)),
        },
        "private_evidence": {"sources": sources, "profiles": profiles},
        "action_vector": {
            "decode_audio": False,
            "transcribe_or_diarize": False,
            "run_models_or_predictions": False,
            "read_pilot_outcome_gold": False,
            "create_or_mutate_identity_records": False,
            "mutate_profiles_or_references": False,
            "write_providers": False,
            "apply_speaker_assignments": False,
            "enable_default_integration": False,
            "run_historical_reprocessing": False,
        },
        "contains_pilot_outcome_gold": False,
        "contains_display_names": False,
    }
    return {**core, "content_sha256": _canonical_hash(core)}


def validate_pilot_proposals(
    value: Mapping[str, Any],
    *,
    expected_speaker_refs: Sequence[str],
    allowlisted_subject_ids: Sequence[str],
) -> dict[str, Any]:
    """Freeze complete proposals while accepting only exact acoustic subject IDs."""

    expected = tuple(str(item) for item in expected_speaker_refs)
    allowlist = tuple(str(item) for item in allowlisted_subject_ids)
    if len(expected) != len(set(expected)) or not expected:
        raise Plan0056PilotError("Expected speaker references are invalid.")
    if len(allowlist) != 2 or len(set(allowlist)) != 2:
        raise Plan0056PilotError("Exactly two allowlisted subject IDs are required.")

    raw_proposals = value.get("proposals")
    if not isinstance(raw_proposals, list) or len(raw_proposals) != len(expected):
        raise Plan0056PilotError("Pilot proposal denominator is incomplete.")

    by_ref: dict[str, dict[str, Any]] = {}
    for raw in raw_proposals:
        if not isinstance(raw, Mapping) or set(raw) != _PROPOSAL_KEYS:
            raise Plan0056PilotError("A pilot proposal has an invalid shape.")
        proposal = dict(raw)
        speaker_ref = str(proposal.get("speaker_ref") or "")
        disposition = proposal.get("disposition")
        subject_id = proposal.get("subject_id")
        if speaker_ref not in expected or speaker_ref in by_ref:
            raise Plan0056PilotError("Pilot proposal references are invalid or duplicated.")
        if disposition not in {"assign", "review", "abstain"}:
            raise Plan0056PilotError("Pilot proposal disposition is invalid.")
        if disposition in {"assign", "review"} and subject_id not in allowlist:
            raise Plan0056PilotError(
                "Non-abstaining proposals must use an exact allowlisted subject ID."
            )
        if disposition == "abstain" and subject_id is not None:
            raise Plan0056PilotError("Abstaining proposals must not carry an identity.")
        if proposal.get("confidence_band") not in {"high", "medium", "low", "none"}:
            raise Plan0056PilotError("Pilot proposal confidence band is invalid.")
        if not " ".join(str(proposal.get("rationale") or "").split()):
            raise Plan0056PilotError("Pilot proposal rationale is required.")
        by_ref[speaker_ref] = proposal

    core = {
        "schema_version": PROPOSAL_SCHEMA,
        "speaker_count": len(expected),
        "allowlisted_subject_ids": list(allowlist),
        "proposals": [by_ref[item] for item in expected],
        "contains_display_names": False,
        "will_apply_assignments": False,
        "requires_human_review": True,
    }
    return {**core, "content_sha256": _canonical_hash(core)}
