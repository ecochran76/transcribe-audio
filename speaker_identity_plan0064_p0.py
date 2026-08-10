from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import stat
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from acoustic_audio_derivatives import (
    canonical_artifact_hash,
    ensure_private_tree,
    read_private_object,
    require_private_file,
    sha256_file,
    utc_now,
    write_immutable_private_json,
)


MANIFEST_SCHEMA = "transcribe-audio.plan0064-p0-manifest.v1"
RECEIPT_SCHEMA = "transcribe-audio.plan0064-p0-receipt.v1"
MODULE_NAME = "speaker_identity_plan0064_p0.py"
DEFAULT_RUNTIME_ROOT = Path("~/.local/state/transcribe-audio/plan-0064")
DEFAULT_TRANSCRIPT_ROOT = Path("~/.transcripts")
DEFAULT_REFERENCE_ROOT = Path(
    "~/.local/state/transcribe-audio/plan-0037/biometric-references"
)
DEFAULT_PROFILE_ROOT = Path(
    "~/.local/state/transcribe-audio/plan-0037/verification-calibration"
)
DEFAULT_PRIOR_CAMPAIGN_ROOT = Path(
    "~/.local/state/transcribe-audio/speaker-evaluation-campaigns"
)
MAX_EVALUATION_RECORDINGS = 12
MAX_PROFILE_MODELS = 3
SHA256_RE = re.compile(r"[a-f0-9]{64}")
COMMIT_RE = re.compile(r"[a-f0-9]{40}")


class Plan0064P0Error(ValueError):
    """Raised when the P0 inventory or cohort cannot fail closed."""

    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


def _fail(reason_code: str, message: str) -> None:
    raise Plan0064P0Error(reason_code, message)


def _canonical_hash(value: Any) -> str:
    return canonical_artifact_hash(value)


def _content_addressed(value: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(value)
    body.pop("content_sha256", None)
    return {**body, "content_sha256": _canonical_hash(body)}


def _json_object(value: Any, *, reason_code: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    try:
        loaded = json.loads(str(value or "{}"))
    except json.JSONDecodeError as exc:
        _fail(reason_code, "A required JSON object is invalid.")
        raise AssertionError from exc
    if not isinstance(loaded, dict):
        _fail(reason_code, "A required JSON value is not an object.")
    return loaded


def _json_list(value: Any, *, reason_code: str) -> list[Any]:
    if isinstance(value, list):
        return list(value)
    try:
        loaded = json.loads(str(value or "[]"))
    except json.JSONDecodeError as exc:
        _fail(reason_code, "A required JSON list is invalid.")
        raise AssertionError from exc
    if not isinstance(loaded, list):
        _fail(reason_code, "A required JSON value is not a list.")
    return loaded


def _readonly_connection(path: Path) -> sqlite3.Connection:
    selected = path.expanduser().absolute()
    if selected.is_symlink() or not selected.is_file():
        _fail("missing_state_database", f"State database is unavailable: {selected}")
    connection = sqlite3.connect(f"file:{selected}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    return connection


def _rows(connection: sqlite3.Connection, table: str) -> list[dict[str, Any]]:
    if not re.fullmatch(r"[a-z][a-z0-9_]*", table):
        _fail("invalid_table_name", "State snapshot table name is invalid.")
    columns = [
        str(row[1])
        for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
    ]
    if not columns:
        _fail("missing_state_table", f"Required state table is missing: {table}")
    ordering = ", ".join(f'"{column}"' for column in columns)
    return [
        {column: row[column] for column in columns}
        for row in connection.execute(
            f'SELECT * FROM "{table}" ORDER BY {ordering}'
        ).fetchall()
    ]


def _database_snapshot(path: Path, tables: Sequence[str]) -> dict[str, Any]:
    with _readonly_connection(path) as connection:
        quick_check = str(connection.execute("PRAGMA quick_check").fetchone()[0])
        table_state = {}
        for table in tables:
            values = _rows(connection, table)
            table_state[table] = {
                "count": len(values),
                "content_sha256": _canonical_hash(values),
            }
    if quick_check != "ok":
        _fail("sqlite_integrity_failed", f"SQLite quick_check failed for {path}.")
    core = {"quick_check": quick_check, "tables": table_state}
    return {**core, "snapshot_sha256": _canonical_hash(core)}


def _regular_file(path: Path, root: Path, *, reason_code: str) -> Path:
    selected_root = root.expanduser().resolve(strict=True)
    selected = path.expanduser().resolve(strict=True)
    try:
        selected.relative_to(selected_root)
    except ValueError as exc:
        _fail(reason_code, "A runtime artifact escapes its governed root.")
        raise AssertionError from exc
    if selected.is_symlink() or not selected.is_file():
        _fail(reason_code, "A runtime artifact is not a regular file.")
    return selected


def _hash_bound_file(
    path: Path,
    root: Path,
    expected_sha256: str,
    *,
    reason_code: str,
) -> dict[str, Any]:
    selected = _regular_file(path, root, reason_code=reason_code)
    actual = sha256_file(selected)
    if not SHA256_RE.fullmatch(str(expected_sha256 or "")) or actual != expected_sha256:
        _fail(reason_code, "A runtime artifact hash does not match its authority.")
    return {
        "path": str(selected),
        "sha256": actual,
        "bytes": selected.stat().st_size,
    }


def _reference_inventory(reference_root: Path) -> dict[str, Any]:
    root = reference_root.expanduser().absolute()
    database = root / "references.sqlite3"
    tables = (
        "profiles",
        "generations",
        "person_heads",
        "events",
        "source_claims",
        "descendants",
    )
    snapshot = _database_snapshot(database, tables)
    with _readonly_connection(database) as connection:
        profiles = _rows(connection, "profiles")
        generations = _rows(connection, "generations")
        heads = _rows(connection, "person_heads")
        claims = _rows(connection, "source_claims")
        descendants = _rows(connection, "descendants")

    active_profiles = [row for row in profiles if row["status"] == "active"]
    active_generations = [
        row
        for row in generations
        if row["status"] == "active" and row["eligible_for_materialization"] == 1
    ]
    active_heads = [row for row in heads if row["status"] == "active"]
    eligible_descendants = [row for row in descendants if row["state"] == "eligible"]
    if not active_profiles or not active_generations:
        _fail("empty_active_reference_inventory", "No active biometric references exist.")

    profile_by_id = {str(row["profile_id"]): row for row in active_profiles}
    generation_by_id = {
        str(row["generation_id"]): row for row in active_generations
    }
    head_by_person = {str(row["person_ref_id"]): row for row in active_heads}
    if (
        len(profile_by_id) != len(active_profiles)
        or len(generation_by_id) != len(active_generations)
        or len(head_by_person) != len(active_heads)
        or set(head_by_person) != {str(row["person_ref_id"]) for row in active_profiles}
    ):
        _fail("active_reference_head_drift", "Active reference heads are incomplete.")

    active_inventory: list[dict[str, Any]] = []
    manifest_sources: list[dict[str, Any]] = []
    for person_ref_id, head in sorted(head_by_person.items()):
        profile = profile_by_id.get(str(head["profile_id"]))
        generation = generation_by_id.get(str(head["generation_id"]))
        if (
            profile is None
            or generation is None
            or profile["person_ref_id"] != person_ref_id
            or profile["head_generation_id"] != generation["generation_id"]
            or generation["profile_id"] != profile["profile_id"]
        ):
            _fail("active_reference_head_drift", "An active reference head drifted.")
        manifest = _json_object(
            generation["manifest_json"], reason_code="invalid_reference_manifest"
        )
        if (
            _canonical_hash(manifest) != generation["manifest_sha256"]
            or manifest.get("generation_id") != generation["generation_id"]
            or manifest.get("person_ref_id") != person_ref_id
            or manifest.get("eligible_for_materialization") is not True
            or _canonical_hash(manifest.get("sources"))
            != manifest.get("source_set_sha256")
        ):
            _fail("invalid_reference_manifest", "An active reference manifest drifted.")
        sources = _json_list(
            manifest.get("sources"), reason_code="invalid_reference_manifest"
        )
        active_inventory.append(
            {
                "person_ref_id": person_ref_id,
                "profile_id": str(profile["profile_id"]),
                "generation_id": str(generation["generation_id"]),
                "generation_sha256": str(generation["manifest_sha256"]),
                "generation_sequence": int(generation["sequence"]),
                "source_count": len(sources),
                "source_set_sha256": str(manifest.get("source_set_sha256") or ""),
                "lifecycle_state": "active",
            }
        )

    all_manifest_sources: list[dict[str, Any]] = []
    for generation in generations:
        manifest = _json_object(
            generation["manifest_json"], reason_code="invalid_reference_manifest"
        )
        if _canonical_hash(manifest) != generation["manifest_sha256"]:
            _fail("invalid_reference_manifest", "A reference manifest hash drifted.")
        for source in _json_list(
            manifest.get("sources"), reason_code="invalid_reference_manifest"
        ):
            if not isinstance(source, Mapping):
                _fail("invalid_reference_manifest", "A reference source is invalid.")
            normalized = dict(source)
            normalized["person_ref_id"] = str(manifest.get("person_ref_id") or "")
            normalized["generation_id"] = str(generation["generation_id"])
            normalized["generation_status"] = str(generation["status"])
            all_manifest_sources.append(normalized)
            if generation in active_generations:
                manifest_sources.append(normalized)

    manifest_claim_keys = {
        (
            str(source.get("source_key") or ""),
            str(source.get("source_sha256") or ""),
            float(source.get("start_seconds") or 0.0),
            float(source.get("end_seconds") or 0.0),
            str(source.get("person_ref_id") or source.get("person_id") or ""),
        )
        for source in all_manifest_sources
    }
    stored_claim_keys = {
        (
            str(row["source_key"]),
            str(row["source_sha256"]),
            float(row["start_seconds"]),
            float(row["end_seconds"]),
            str(row["person_ref_id"]),
        )
        for row in claims
    }
    if manifest_claim_keys != stored_claim_keys:
        _fail("reference_source_claim_drift", "Reference source claims drifted.")

    descendant_by_id = {str(row["descendant_id"]): row for row in descendants}
    if len(descendant_by_id) != len(descendants):
        _fail("duplicate_reference_descendant", "Reference descendants are duplicated.")
    return {
        "database": str(database),
        "snapshot": snapshot,
        "active_references": active_inventory,
        "active_reference_set_sha256": _canonical_hash(active_inventory),
        "eligible_descendant_ids": sorted(
            str(row["descendant_id"]) for row in eligible_descendants
        ),
        "eligible_descendants": sorted(
            eligible_descendants, key=lambda row: str(row["descendant_id"])
        ),
        "descendant_count_by_state": dict(
            sorted(Counter(str(row["state"]) for row in descendants).items())
        ),
        "generation_count_by_state": dict(
            sorted(Counter(str(row["status"]) for row in generations).items())
        ),
        "profile_count_by_state": dict(
            sorted(Counter(str(row["status"]) for row in profiles).items())
        ),
        "development_sources": sorted(
            all_manifest_sources,
            key=lambda item: (
                str(item.get("source_sha256") or ""),
                float(item.get("start_seconds") or 0.0),
                str(item.get("person_ref_id") or item.get("person_id") or ""),
                str(item.get("reference_id") or ""),
            ),
        ),
        "active_generation_source_count": len(manifest_sources),
        "historical_source_claim_count": len(claims),
        "development_recording_hashes": sorted(
            {str(row["source_sha256"]) for row in claims}
        ),
    }


def _profile_inventory(
    profile_root: Path, reference_inventory: Mapping[str, Any]
) -> dict[str, Any]:
    root = profile_root.expanduser().absolute()
    database = root / "profiles.sqlite3"
    snapshot = _database_snapshot(database, ("profiles",))
    with _readonly_connection(database) as connection:
        profiles = _rows(connection, "profiles")
    active = [row for row in profiles if row["lifecycle_state"] == "active"]
    if not active:
        _fail("empty_active_profile_inventory", "No active model profiles exist.")
    candidate_ids = sorted({str(row["candidate_id"]) for row in active})
    if not candidate_ids or len(candidate_ids) > MAX_PROFILE_MODELS:
        _fail("profile_model_bound_exceeded", "The active model bound is invalid.")
    eligible_descendants = set(reference_inventory["eligible_descendant_ids"])
    active_reference_ids = {
        str(row["profile_id"])
        for row in reference_inventory["active_references"]
    }
    active_generation_ids = {
        str(row["generation_id"])
        for row in reference_inventory["active_references"]
    }
    reference_by_generation = {
        str(row["generation_id"]): row
        for row in reference_inventory["active_references"]
    }
    descendant_by_id = {
        str(row["descendant_id"]): row
        for row in reference_inventory["eligible_descendants"]
    }
    normalized: list[dict[str, Any]] = []
    seen_profile_ids: set[str] = set()
    seen_subject_models: set[tuple[str, str]] = set()
    for row in active:
        profile_id = str(row["profile_id"])
        subject_model = (str(row["person_ref_id"]), str(row["candidate_id"]))
        if profile_id in seen_profile_ids or subject_model in seen_subject_models:
            _fail("duplicate_active_profile", "An active model profile is duplicated.")
        seen_profile_ids.add(profile_id)
        seen_subject_models.add(subject_model)
        if (
            str(row["descendant_id"]) not in eligible_descendants
            or str(row["p3_profile_id"]) not in active_reference_ids
            or str(row["generation_id"]) not in active_generation_ids
        ):
            _fail(
                "profile_reference_binding_drift",
                "An active model profile lost its eligible reference binding.",
            )
        reference = reference_by_generation[str(row["generation_id"])]
        descendant = descendant_by_id[str(row["descendant_id"])]
        if (
            str(row["generation_sha256"]) != reference["generation_sha256"]
            or str(descendant["generation_id"]) != row["generation_id"]
            or str(descendant["profile_id"]) != row["p3_profile_id"]
            or str(descendant["generation_sha256"]) != row["generation_sha256"]
            or str(descendant["artifact_sha256"]) != row["artifact_sha256"]
        ):
            _fail(
                "profile_reference_lineage_drift",
                "An active profile's reference lineage drifted.",
            )
        artifact = _hash_bound_file(
            Path(str(row["artifact_path"])),
            root,
            str(row["artifact_sha256"]),
            reason_code="profile_artifact_drift",
        )
        manifest = _hash_bound_file(
            Path(str(row["profile_manifest_path"])),
            root,
            str(row["profile_manifest_sha256"]),
            reason_code="profile_manifest_drift",
        )
        profile_manifest = read_private_object(Path(manifest["path"]))
        expected_manifest_fields = {
            "profile_id": row["profile_id"],
            "descendant_id": row["descendant_id"],
            "person_ref_id": row["person_ref_id"],
            "p3_profile_id": row["p3_profile_id"],
            "generation_id": row["generation_id"],
            "generation_sha256": row["generation_sha256"],
            "candidate_id": row["candidate_id"],
            "model_revision": row["model_revision"],
            "preprocessing": _json_object(
                row["preprocessing_json"], reason_code="invalid_profile_metadata"
            ),
            "artifact_path": row["artifact_path"],
            "artifact_sha256": row["artifact_sha256"],
            "vector_dimension": row["vector_dimension"],
            "window_count": row["window_count"],
            "session_count": row["session_count"],
            "dispersion": row["dispersion"],
        }
        if (
            profile_manifest.get("contains_raw_biometric_values") is not False
            or any(
                profile_manifest.get(key) != value
                for key, value in expected_manifest_fields.items()
            )
        ):
            _fail("profile_manifest_drift", "A profile manifest binding drifted.")
        lifecycle_sha256 = str(row["state_receipt_sha256"] or "")
        if not SHA256_RE.fullmatch(lifecycle_sha256):
            _fail("profile_lifecycle_receipt_drift", "A lifecycle receipt is missing.")
        lifecycle_path = root / "authority" / f"{lifecycle_sha256}.json"
        try:
            require_private_file(lifecycle_path, root)
            lifecycle = read_private_object(lifecycle_path)
        except (OSError, ValueError) as exc:
            _fail(
                "profile_lifecycle_receipt_drift",
                "A lifecycle receipt is unavailable.",
            )
            raise AssertionError from exc
        if (
            _canonical_hash(lifecycle) != lifecycle_sha256
            or lifecycle.get("profile_id") != row["profile_id"]
            or lifecycle.get("descendant_id") != row["descendant_id"]
            or lifecycle.get("artifact_sha256") != row["artifact_sha256"]
            or lifecycle.get("profile_manifest_sha256")
            != row["profile_manifest_sha256"]
            or lifecycle.get("to_state") != row["lifecycle_state"]
            or lifecycle.get("replacement_profile_id")
            != row["replacement_profile_id"]
            or lifecycle.get("will_perform_external_write") is not False
        ):
            _fail(
                "profile_lifecycle_receipt_drift",
                "A lifecycle receipt binding drifted.",
            )
        normalized.append(
            {
                "profile_id": profile_id,
                "person_ref_id": str(row["person_ref_id"]),
                "descendant_id": str(row["descendant_id"]),
                "reference_profile_id": str(row["p3_profile_id"]),
                "generation_id": str(row["generation_id"]),
                "generation_sha256": str(row["generation_sha256"]),
                "candidate_id": str(row["candidate_id"]),
                "model_revision": str(row["model_revision"]),
                "preprocessing": _json_object(
                    row["preprocessing_json"], reason_code="invalid_profile_metadata"
                ),
                "artifact": artifact,
                "manifest": manifest,
                "state_receipt_sha256": lifecycle_sha256,
                "window_count": int(row["window_count"]),
                "session_count": int(row["session_count"]),
                "lifecycle_state": "active",
            }
        )
    normalized.sort(key=lambda item: (item["person_ref_id"], item["candidate_id"]))
    subject_counts = Counter(item["person_ref_id"] for item in normalized)
    if any(count != len(candidate_ids) for count in subject_counts.values()):
        _fail("incomplete_subject_model_matrix", "The active subject/model matrix is incomplete.")
    return {
        "database": str(database),
        "snapshot": snapshot,
        "active_profiles": normalized,
        "active_profile_set_sha256": _canonical_hash(normalized),
        "candidate_ids": candidate_ids,
        "model_revisions": {
            candidate_id: sorted(
                {
                    item["model_revision"]
                    for item in normalized
                    if item["candidate_id"] == candidate_id
                }
            )
            for candidate_id in candidate_ids
        },
        "profile_count_by_state": dict(
            sorted(Counter(str(row["lifecycle_state"]) for row in profiles).items())
        ),
        "subject_count": len(subject_counts),
    }


def _canonical_bindings(
    transcript_root: Path, profile_inventory: Mapping[str, Any]
) -> dict[str, Any]:
    root = transcript_root.expanduser().absolute()
    database = root / "transcripts.sqlite3"
    tables = (
        "knowledge_people",
        "knowledge_source_records",
        "knowledge_observations",
        "knowledge_current_person_profiles",
    )
    snapshot = _database_snapshot(database, tables)
    with _readonly_connection(database) as connection:
        people = _rows(connection, "knowledge_people")
        source_records = _rows(connection, "knowledge_source_records")
        observations = _rows(connection, "knowledge_observations")
        person_profiles = _rows(connection, "knowledge_current_person_profiles")
    people_by_id = {str(row["id"]): row for row in people}
    if len(people_by_id) != len(people):
        _fail("duplicate_canonical_person", "Canonical people are duplicated.")
    explicit: dict[str, str] = {}
    binding_observations: list[dict[str, Any]] = []
    for observation in observations:
        if (
            observation["observation_type"]
            != "reviewed_voice_subject_binding_confirmed"
            or observation["review_state"] != "accepted"
        ):
            continue
        payload = _json_object(
            observation["payload_json"], reason_code="invalid_voice_person_binding"
        )
        if payload.get("active_binding") is not True:
            continue
        acoustic_subject_id = str(payload.get("acoustic_subject_id") or "")
        person_id = str(payload.get("person_id") or observation["subject_id"] or "")
        if not acoustic_subject_id or person_id not in people_by_id:
            _fail("invalid_voice_person_binding", "A reviewed voice binding is invalid.")
        if acoustic_subject_id in explicit and explicit[acoustic_subject_id] != person_id:
            _fail("conflicting_voice_person_binding", "Voice bindings conflict.")
        explicit[acoustic_subject_id] = person_id
        binding_observations.append(
            {
                "observation_id": str(observation["id"]),
                "acoustic_subject_id": acoustic_subject_id,
                "person_id": person_id,
                "content_hash": str(observation["content_hash"]),
                "observed_at": str(observation["observed_at"]),
            }
        )

    subject_ids = sorted(
        {str(row["person_ref_id"]) for row in profile_inventory["active_profiles"]}
    )
    bindings: list[dict[str, Any]] = []
    for subject_id in subject_ids:
        if subject_id in people_by_id:
            person_id = subject_id
            status = "direct_canonical_person_id"
            observation_id = ""
        elif subject_id in explicit:
            person_id = explicit[subject_id]
            status = "accepted_explicit_voice_person_binding"
            observation_id = next(
                row["observation_id"]
                for row in binding_observations
                if row["acoustic_subject_id"] == subject_id
            )
        else:
            person_id = ""
            status = "missing_canonical_person_binding"
            observation_id = ""
        bindings.append(
            {
                "acoustic_subject_id": subject_id,
                "person_id": person_id,
                "binding_status": status,
                "binding_observation_id": observation_id,
                "identity_candidate_eligible": bool(person_id),
            }
        )

    person_ids = {item["person_id"] for item in bindings if item["person_id"]}
    affinities = [
        {
            key: row[key]
            for key in row
            if key
            not in {
                "label",
                "normalized_value",
                "display_value",
            }
        }
        for row in source_records
        if str(row.get("person_id") or "") in person_ids
    ]
    current_profiles = [
        row for row in person_profiles if str(row.get("person_id") or "") in person_ids
    ]
    return {
        "database": str(database),
        "snapshot": snapshot,
        "canonical_person_count": len(people),
        "subject_bindings": bindings,
        "binding_set_sha256": _canonical_hash(bindings),
        "accepted_explicit_binding_observations": sorted(
            binding_observations, key=lambda item: item["observation_id"]
        ),
        "canonical_source_affinities": sorted(
            affinities, key=lambda item: str(item.get("id") or "")
        ),
        "current_person_profiles": sorted(
            current_profiles, key=lambda item: str(item.get("person_id") or "")
        ),
        "binding_status_counts": dict(
            sorted(Counter(item["binding_status"] for item in bindings).items())
        ),
        "identity_ready_subject_count": sum(
            item["identity_candidate_eligible"] for item in bindings
        ),
        "unbound_subject_count": sum(
            not item["identity_candidate_eligible"] for item in bindings
        ),
    }


def _strings_at_key(value: Any, key: str) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for current_key, current_value in value.items():
            if current_key == key and isinstance(current_value, str) and current_value:
                found.add(current_value)
            found.update(_strings_at_key(current_value, key))
    elif isinstance(value, list):
        for item in value:
            found.update(_strings_at_key(item, key))
    return found


def _prior_exposure(prior_campaign_root: Path | None) -> dict[str, Any]:
    if prior_campaign_root is None:
        return {
            "root": "",
            "document_ids": [],
            "evidence": [],
            "exposure_set_sha256": _canonical_hash([]),
        }
    root = prior_campaign_root.expanduser().absolute()
    if not root.exists():
        return {
            "root": str(root),
            "document_ids": [],
            "evidence": [],
            "exposure_set_sha256": _canonical_hash([]),
        }
    if root.is_symlink() or not root.is_dir():
        _fail("unsafe_prior_campaign_root", "Prior campaign root is unsafe.")
    document_ids: set[str] = set()
    evidence: list[dict[str, Any]] = []
    for gold_root in sorted(root.glob("campaign-*/gold")):
        if gold_root.is_symlink() or not gold_root.is_dir():
            _fail("unsafe_prior_exposure", "Prior gold directory is unsafe.")
        for child in sorted(gold_root.iterdir()):
            if child.is_dir() and not child.is_symlink():
                document_ids.add(child.name)
                evidence.append(
                    {
                        "kind": "gold_directory_presence_only",
                        "path": str(child),
                        "document_id": child.name,
                    }
                )
    for review_path in sorted(root.glob("campaign-*/review-opens/*.json")):
        name = review_path.stem
        if "-" in name:
            document_id = name.split("-", 1)[1]
            document_ids.add(document_id)
            evidence.append(
                {
                    "kind": "review_open_filename_only",
                    "path": str(review_path),
                    "document_id": document_id,
                    "sha256": sha256_file(review_path),
                }
            )
    exposure_dirs = ("freezes", "baselines", "reruns")
    for directory in exposure_dirs:
        for path in sorted(root.glob(f"campaign-*/{directory}/**/*.json")):
            if path.is_symlink() or not path.is_file():
                _fail("unsafe_prior_exposure", "Prior exposure evidence is unsafe.")
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                _fail("invalid_prior_exposure", "Prior exposure evidence is invalid.")
                raise AssertionError from exc
            ids = sorted(_strings_at_key(value, "document_id"))
            document_ids.update(ids)
            evidence.append(
                {
                    "kind": f"{directory}_document_ids",
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "document_ids": ids,
                }
            )
    evidence.sort(key=lambda item: (item["kind"], item["path"]))
    return {
        "root": str(root),
        "document_ids": sorted(document_ids),
        "evidence": evidence,
        "exposure_set_sha256": _canonical_hash(
            {"document_ids": sorted(document_ids), "evidence": evidence}
        ),
    }


def _recording_time(row: Mapping[str, Any], payload: Mapping[str, Any]) -> tuple[str, datetime]:
    candidates = (
        payload.get("recording_start"),
        payload.get("recording_end"),
        row.get("generated_at"),
        row.get("created_at"),
    )
    for value in candidates:
        text = str(value or "").strip()
        if not text:
            continue
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return text, parsed.astimezone(timezone.utc)
    return "", datetime.max.replace(tzinfo=timezone.utc)


def _candidate_rows(transcript_root: Path) -> tuple[list[dict[str, Any]], int]:
    root = transcript_root.expanduser().absolute()
    database = root / "transcripts.sqlite3"
    with _readonly_connection(database) as connection:
        rows = connection.execute(
            """
            SELECT d.*, b.id AS media_blob_id, b.stored_path AS media_stored_path,
                   b.sha256 AS media_sha256, b.bytes AS media_bytes,
                   b.mime_type AS media_mime_type
            FROM documents d
            LEFT JOIN document_blobs db
              ON db.document_id = d.id AND db.role = 'source_recording'
            LEFT JOIN blobs b ON b.id = db.blob_id
            WHERE d.kind = 'transcript'
            ORDER BY d.id
            """
        ).fetchall()
    prepared: list[tuple[datetime, str, dict[str, Any]]] = []
    for row in rows:
        item = {key: row[key] for key in row.keys()}
        payload = _json_object(
            row["json_payload"], reason_code="invalid_transcript_document"
        )
        recording_time, sort_time = _recording_time(item, payload)
        prepared.append(
            (
                sort_time,
                str(row["id"]),
                {"row": item, "payload": payload, "recording_time": recording_time},
            )
        )
    prepared.sort(key=lambda item: (item[0], item[1]))
    result = [item[2] for item in prepared]
    for rank, item in enumerate(result, start=1):
        item["chronological_rank"] = rank
    return result, len(rows)


def _cohort(
    transcript_root: Path,
    *,
    development_hashes: Iterable[str],
    prior_exposure: Mapping[str, Any],
    limit: int,
) -> dict[str, Any]:
    if limit < 1 or limit > MAX_EVALUATION_RECORDINGS:
        _fail("evaluation_bound_exceeded", "Evaluation cohort bound is invalid.")
    root = transcript_root.expanduser().absolute()
    candidates, total_count = _candidate_rows(root)
    excluded_hashes = set(development_hashes)
    exposed_documents = set(prior_exposure["document_ids"])
    seen_media_hashes: dict[str, str] = {}
    considered: list[dict[str, Any]] = []
    selected_count = 0
    for candidate in candidates:
        row = candidate["row"]
        payload = candidate["payload"]
        document_id = str(row["id"])
        media_sha256 = str(row.get("media_sha256") or "")
        utterances = (
            payload.get("utterances")
            if isinstance(payload.get("utterances"), list)
            else []
        )
        speaker_labels = sorted(
            {
                str(item.get("speaker") or "").strip()
                for item in utterances
                if isinstance(item, Mapping) and str(item.get("speaker") or "").strip()
            }
        )
        reason_codes: list[str] = []
        transcript_artifact: dict[str, Any] | None = None
        media_artifact: dict[str, Any] | None = None
        if document_id in exposed_documents:
            reason_codes.append("prior_identity_evidence_exposure")
        if media_sha256 and media_sha256 in excluded_hashes:
            reason_codes.append("development_recording_overlap")
        if media_sha256 and media_sha256 in seen_media_hashes:
            reason_codes.append("repeated_recording_hash")
        if not candidate["recording_time"]:
            reason_codes.append("missing_recording_time")
        if not utterances:
            reason_codes.append("missing_diarization")
        elif len(speaker_labels) < 2:
            reason_codes.append("fewer_than_two_speaker_labels")
        try:
            transcript_artifact = _hash_bound_file(
                Path(str(row["stored_path"])),
                root,
                str(row["artifact_sha256"]),
                reason_code="transcript_artifact_unavailable",
            )
        except (OSError, Plan0064P0Error):
            reason_codes.append("transcript_artifact_unavailable")
        if not media_sha256 or not row.get("media_stored_path"):
            reason_codes.append("source_media_unavailable")
        else:
            try:
                media_artifact = _hash_bound_file(
                    Path(str(row["media_stored_path"])),
                    root,
                    media_sha256,
                    reason_code="source_media_unavailable",
                )
            except (OSError, Plan0064P0Error):
                reason_codes.append("source_media_unavailable")
        event = payload.get("event")
        if isinstance(event, Mapping):
            context_status = "local_calendar_context_available"
        elif payload.get("conversation_id") and payload.get("recording_id"):
            context_status = "durable_identity_context_available"
        else:
            context_status = "transcript_seed_only_requires_p2_retrieval"
        structural_reasons = sorted(set(reason_codes))
        eligible = not structural_reasons
        if eligible:
            selected_count += 1
            disposition = "selected_evaluation_candidate"
        else:
            disposition = "excluded"
        considered.append(
            {
                "chronological_rank": int(candidate["chronological_rank"]),
                "document_id": document_id,
                "recording_time": candidate["recording_time"],
                "artifact_sha256": str(row["artifact_sha256"]),
                "recording_id": str(payload.get("recording_id") or ""),
                "conversation_id": str(payload.get("conversation_id") or ""),
                "source_media_sha256": media_sha256,
                "source_media_blob_id": str(row.get("media_blob_id") or ""),
                "source_media_mime_type": str(row.get("media_mime_type") or ""),
                "transcript_artifact": transcript_artifact,
                "source_media_artifact": media_artifact,
                "utterance_count": len(utterances),
                "speaker_labels": speaker_labels,
                "context_status": context_status,
                "eligible": eligible,
                "disposition": disposition,
                "reason_codes": structural_reasons or ["eligible"],
            }
        )
        if media_sha256:
            seen_media_hashes.setdefault(media_sha256, document_id)
        if selected_count == limit:
            break
    if selected_count < limit:
        _fail(
            "incomplete_candidate_denominator",
            f"Only {selected_count} of {limit} required evaluation recordings are eligible.",
        )
    selected = [item for item in considered if item["eligible"]]
    core = {
        "selection_policy": {
            "order": "oldest_recording_time_then_document_id",
            "max_evaluation_recordings": limit,
            "requires_hash_matched_transcript": True,
            "requires_hash_matched_source_media": True,
            "requires_two_diarized_labels": True,
            "excludes_development_recording_hashes": True,
            "excludes_prior_identity_evidence_exposure": True,
            "calendar_absence_alone_excludes": False,
        },
        "total_transcript_documents": total_count,
        "considered_count": len(considered),
        "selected_count": len(selected),
        "last_considered_chronological_rank": considered[-1]["chronological_rank"],
        "disposition_counts": dict(
            sorted(Counter(item["disposition"] for item in considered).items())
        ),
        "reason_code_counts": dict(
            sorted(
                Counter(
                    reason
                    for item in considered
                    for reason in item["reason_codes"]
                ).items()
            )
        ),
        "considered": considered,
        "selected_document_ids": [item["document_id"] for item in selected],
        "selected_recording_hashes": [
            item["source_media_sha256"] for item in selected
        ],
    }
    return {**core, "cohort_sha256": _canonical_hash(core)}


def _git(args: Sequence[str], *, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", *args],
        cwd=Path(__file__).resolve().parent,
        capture_output=True,
        text=not binary,
        check=False,
    )
    if result.returncode:
        _fail("repository_authority_unavailable", "Repository authority is unavailable.")
    return result.stdout if binary else result.stdout.strip()


def repository_authority(*, require_clean: bool) -> dict[str, Any]:
    status = str(_git(["status", "--porcelain=v1", "--untracked-files=normal"]))
    counts = str(
        _git(["rev-list", "--left-right", "--count", "@{upstream}...HEAD"])
    ).split()
    commit = str(_git(["log", "-1", "--format=%H", "--", MODULE_NAME]))
    module_sha256 = sha256_file(Path(__file__).resolve())
    blob_matches = False
    if COMMIT_RE.fullmatch(commit):
        blob = _git(["show", f"{commit}:{MODULE_NAME}"], binary=True)
        blob_matches = isinstance(blob, bytes) and hashlib.sha256(blob).hexdigest() == module_sha256
    authority = {
        "head": str(_git(["rev-parse", "HEAD"])),
        "module_commit": commit,
        "module_name": MODULE_NAME,
        "module_sha256": module_sha256,
        "module_blob_matches": blob_matches,
        "clean": not status,
        "upstream_behind": int(counts[0]) if len(counts) == 2 else -1,
        "upstream_ahead": int(counts[1]) if len(counts) == 2 else -1,
    }
    if require_clean and (
        authority["clean"] is not True
        or authority["upstream_behind"] != 0
        or authority["upstream_ahead"] != 0
        or authority["module_blob_matches"] is not True
    ):
        _fail(
            "repository_authority_drift",
            "P0 freeze requires a clean upstream-even committed module.",
        )
    return authority


def build_p0_manifest(
    *,
    transcript_root: Path = DEFAULT_TRANSCRIPT_ROOT,
    reference_root: Path = DEFAULT_REFERENCE_ROOT,
    profile_root: Path = DEFAULT_PROFILE_ROOT,
    prior_campaign_root: Path | None = DEFAULT_PRIOR_CAMPAIGN_ROOT,
    evaluation_limit: int = MAX_EVALUATION_RECORDINGS,
    repository: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the complete read-only P0 inventory and chronological cohort."""

    references = _reference_inventory(reference_root)
    profiles = _profile_inventory(profile_root, references)
    bindings = _canonical_bindings(transcript_root, profiles)
    exposure = _prior_exposure(prior_campaign_root)
    cohort = _cohort(
        transcript_root,
        development_hashes=references["development_recording_hashes"],
        prior_exposure=exposure,
        limit=evaluation_limit,
    )
    binding_by_subject = {
        item["acoustic_subject_id"]: item for item in bindings["subject_bindings"]
    }
    active_profiles = []
    for profile in profiles["active_profiles"]:
        binding = binding_by_subject[str(profile["person_ref_id"])]
        active_profiles.append(
            {
                **profile,
                "canonical_person_id": binding["person_id"],
                "binding_status": binding["binding_status"],
                "identity_candidate_eligible": binding[
                    "identity_candidate_eligible"
                ],
            }
        )
    profiles = {**profiles, "active_profiles": active_profiles}
    profiles["identity_ready_profile_count"] = sum(
        item["identity_candidate_eligible"] for item in active_profiles
    )
    profiles["unbound_active_profile_count"] = sum(
        not item["identity_candidate_eligible"] for item in active_profiles
    )
    profiles["active_profile_set_sha256"] = _canonical_hash(active_profiles)
    action_counts = {
        "speaker_assignments": 0,
        "new_enrollments": 0,
        "provider_writes": 0,
        "graphiti_writes": 0,
        "historical_reprocessing": 0,
        "knowledge_writes": 0,
        "external_writes": 0,
    }
    core = {
        "schema_version": MANIFEST_SCHEMA,
        "status": "p0_inventory_and_cohort_ready_to_freeze",
        "repository_authority": dict(repository or repository_authority(require_clean=False)),
        "reference_inventory": references,
        "profile_inventory": profiles,
        "canonical_bindings": bindings,
        "prior_identity_exposure": exposure,
        "evaluation_cohort": cohort,
        "action_counts": action_counts,
        "will_score_speakers": False,
        "will_read_gold": False,
        "will_mutate_runtime_state": False,
        "will_perform_external_write": False,
    }
    return _content_addressed(core)


def _paths(runtime_root: Path, content_sha256: str) -> dict[str, Path]:
    root = runtime_root.expanduser().absolute()
    run = root / f"p0-{content_sha256[:24]}"
    return {
        "root": root,
        "run": run,
        "manifest": run / "private-manifest.json",
        "receipt": run / "receipt.json",
    }


def _validate_manifest(manifest: Mapping[str, Any]) -> str:
    value = dict(manifest)
    expected = _content_addressed(value)
    if (
        value != expected
        or value.get("schema_version") != MANIFEST_SCHEMA
        or value.get("status") != "p0_inventory_and_cohort_ready_to_freeze"
        or value.get("will_score_speakers") is not False
        or value.get("will_read_gold") is not False
        or value.get("will_mutate_runtime_state") is not False
        or value.get("will_perform_external_write") is not False
        or any(value.get("action_counts", {}).values())
    ):
        _fail("invalid_p0_manifest", "The P0 manifest contract is invalid.")
    return str(value["content_sha256"])


def freeze_p0(
    *,
    expected_content_sha256: str,
    transcript_root: Path = DEFAULT_TRANSCRIPT_ROOT,
    reference_root: Path = DEFAULT_REFERENCE_ROOT,
    profile_root: Path = DEFAULT_PROFILE_ROOT,
    prior_campaign_root: Path | None = DEFAULT_PRIOR_CAMPAIGN_ROOT,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    evaluation_limit: int = MAX_EVALUATION_RECORDINGS,
) -> dict[str, Any]:
    """Freeze one exact zero-effect P0 manifest or replay it idempotently."""

    if not SHA256_RE.fullmatch(expected_content_sha256):
        _fail("invalid_expected_hash", "Expected P0 content hash is invalid.")
    authority = repository_authority(require_clean=True)
    manifest = build_p0_manifest(
        transcript_root=transcript_root,
        reference_root=reference_root,
        profile_root=profile_root,
        prior_campaign_root=prior_campaign_root,
        evaluation_limit=evaluation_limit,
        repository=authority,
    )
    content_sha256 = _validate_manifest(manifest)
    if content_sha256 != expected_content_sha256:
        _fail("p0_preview_drift", "The P0 preview changed before freeze.")
    paths = _paths(runtime_root, content_sha256)
    if paths["manifest"].exists() or paths["receipt"].exists():
        if not paths["manifest"].exists() or not paths["receipt"].exists():
            _fail("partial_p0_freeze", "A partial P0 freeze exists.")
        return replay_p0(
            content_sha256=content_sha256,
            transcript_root=transcript_root,
            reference_root=reference_root,
            profile_root=profile_root,
            prior_campaign_root=prior_campaign_root,
            runtime_root=runtime_root,
            evaluation_limit=evaluation_limit,
        )
    ensure_private_tree(paths["root"], paths["run"])
    write_immutable_private_json(paths["manifest"], manifest)
    receipt = _content_addressed(
        {
            "schema_version": RECEIPT_SCHEMA,
            "status": "p0_frozen_zero_effect",
            "manifest_content_sha256": content_sha256,
            "manifest_file_sha256": sha256_file(paths["manifest"]),
            "frozen_at": utc_now(),
            "active_reference_count": len(
                manifest["reference_inventory"]["active_references"]
            ),
            "active_profile_count": len(
                manifest["profile_inventory"]["active_profiles"]
            ),
            "identity_ready_profile_count": manifest["profile_inventory"][
                "identity_ready_profile_count"
            ],
            "unbound_active_profile_count": manifest["profile_inventory"][
                "unbound_active_profile_count"
            ],
            "development_recording_hash_count": len(
                manifest["reference_inventory"]["development_recording_hashes"]
            ),
            "historical_source_claim_count": manifest["reference_inventory"][
                "historical_source_claim_count"
            ],
            "considered_recording_count": manifest["evaluation_cohort"][
                "considered_count"
            ],
            "selected_recording_count": manifest["evaluation_cohort"][
                "selected_count"
            ],
            "action_counts": dict(manifest["action_counts"]),
            "will_perform_external_write": False,
        }
    )
    write_immutable_private_json(paths["receipt"], receipt)
    return {
        "status": "p0_frozen_zero_effect",
        "content_sha256": content_sha256,
        "receipt_content_sha256": receipt["content_sha256"],
        "private_manifest_path": str(paths["manifest"]),
        "private_receipt_path": str(paths["receipt"]),
        "summary": _public_summary(manifest),
        "idempotent_replay": False,
    }


def replay_p0(
    *,
    content_sha256: str,
    transcript_root: Path = DEFAULT_TRANSCRIPT_ROOT,
    reference_root: Path = DEFAULT_REFERENCE_ROOT,
    profile_root: Path = DEFAULT_PROFILE_ROOT,
    prior_campaign_root: Path | None = DEFAULT_PRIOR_CAMPAIGN_ROOT,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    evaluation_limit: int = MAX_EVALUATION_RECORDINGS,
) -> dict[str, Any]:
    """Replay the frozen P0 manifest against the unchanged live read state."""

    if not SHA256_RE.fullmatch(content_sha256):
        _fail("invalid_p0_hash", "P0 content hash is invalid.")
    paths = _paths(runtime_root, content_sha256)
    require_private_file(paths["manifest"], paths["root"])
    require_private_file(paths["receipt"], paths["root"])
    manifest = read_private_object(paths["manifest"])
    receipt = read_private_object(paths["receipt"])
    if _validate_manifest(manifest) != content_sha256:
        _fail("p0_manifest_drift", "The frozen P0 manifest drifted.")
    expected = build_p0_manifest(
        transcript_root=transcript_root,
        reference_root=reference_root,
        profile_root=profile_root,
        prior_campaign_root=prior_campaign_root,
        evaluation_limit=evaluation_limit,
        repository=manifest["repository_authority"],
    )
    if expected != manifest:
        _fail("p0_live_state_drift", "Live P0 inputs changed after freeze.")
    if (
        receipt.get("schema_version") != RECEIPT_SCHEMA
        or receipt.get("status") != "p0_frozen_zero_effect"
        or receipt.get("manifest_content_sha256") != content_sha256
        or receipt.get("manifest_file_sha256") != sha256_file(paths["manifest"])
        or any(receipt.get("action_counts", {}).values())
        or receipt.get("will_perform_external_write") is not False
        or _content_addressed(receipt) != receipt
    ):
        _fail("p0_receipt_drift", "The frozen P0 receipt drifted.")
    return {
        "status": "p0_frozen_zero_effect",
        "content_sha256": content_sha256,
        "receipt_content_sha256": receipt["content_sha256"],
        "private_manifest_path": str(paths["manifest"]),
        "private_receipt_path": str(paths["receipt"]),
        "summary": _public_summary(manifest),
        "idempotent_replay": True,
    }


def _public_summary(manifest: Mapping[str, Any]) -> dict[str, Any]:
    references = manifest["reference_inventory"]
    profiles = manifest["profile_inventory"]
    bindings = manifest["canonical_bindings"]
    cohort = manifest["evaluation_cohort"]
    return {
        "active_reference_count": len(references["active_references"]),
        "active_profile_count": len(profiles["active_profiles"]),
        "active_subject_count": profiles["subject_count"],
        "identity_ready_subject_count": bindings["identity_ready_subject_count"],
        "unbound_subject_count": bindings["unbound_subject_count"],
        "identity_ready_profile_count": profiles["identity_ready_profile_count"],
        "unbound_active_profile_count": profiles["unbound_active_profile_count"],
        "profile_model_count": len(profiles["candidate_ids"]),
        "historical_source_claim_count": references[
            "historical_source_claim_count"
        ],
        "development_recording_hash_count": len(
            references["development_recording_hashes"]
        ),
        "prior_exposed_document_count": len(
            manifest["prior_identity_exposure"]["document_ids"]
        ),
        "total_transcript_documents": cohort["total_transcript_documents"],
        "considered_recording_count": cohort["considered_count"],
        "selected_recording_count": cohort["selected_count"],
        "last_considered_chronological_rank": cohort[
            "last_considered_chronological_rank"
        ],
        "reason_code_counts": cohort["reason_code_counts"],
        "action_counts": dict(manifest["action_counts"]),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Freeze or replay Plan 0064 P0.")
    parser.add_argument("action", choices=("preview", "freeze", "replay"))
    parser.add_argument("--content-sha256", default="")
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--transcript-root", type=Path, default=DEFAULT_TRANSCRIPT_ROOT)
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--profile-root", type=Path, default=DEFAULT_PROFILE_ROOT)
    parser.add_argument(
        "--prior-campaign-root", type=Path, default=DEFAULT_PRIOR_CAMPAIGN_ROOT
    )
    parser.add_argument(
        "--evaluation-limit", type=int, default=MAX_EVALUATION_RECORDINGS
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    common = {
        "transcript_root": args.transcript_root,
        "reference_root": args.reference_root,
        "profile_root": args.profile_root,
        "prior_campaign_root": args.prior_campaign_root,
        "evaluation_limit": args.evaluation_limit,
    }
    if args.action == "preview":
        manifest = build_p0_manifest(**common)
        result = {
            "status": manifest["status"],
            "content_sha256": manifest["content_sha256"],
            "summary": _public_summary(manifest),
        }
    elif args.action == "freeze":
        result = freeze_p0(
            expected_content_sha256=args.content_sha256,
            runtime_root=args.runtime_root,
            **common,
        )
    else:
        result = replay_p0(
            content_sha256=args.content_sha256,
            runtime_root=args.runtime_root,
            **common,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
