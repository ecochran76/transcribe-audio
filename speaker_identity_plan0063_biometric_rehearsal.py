"""Rehearse Plan 0063 biometric references and profiles on private copies.

The module has no live apply entry point. It copies the governed reference and
profile state, registers only the sources selected by a complete Plan 0063 P4
transition, materializes model profiles on those copies, exercises the normal
withdraw/delete lifecycle, and finally restores the exact copied baseline.
"""

from __future__ import annotations

import os
import shutil
import stat
import struct
import wave
from pathlib import Path
from typing import Any, Mapping, Sequence

import acoustic_biometric_references as references
from acoustic_audio_derivatives import (
    AudioDerivativeError,
    canonical_artifact_hash,
    ensure_private_tree,
    read_private_object,
    require_private_file,
    resolve_active_derivative,
    resolve_derivative_lineage_receipt,
    sha256_file,
    write_immutable_private_json,
)
import acoustic_verification as verification
import speaker_identity_orchestration as orchestration
import speaker_identity_plan0063_private_rehearsal as canonical_rehearsal


BIOMETRIC_REHEARSAL_SCHEMA = (
    "transcribe-audio.plan0063-biometric-private-copy-rehearsal.v1"
)
BIOMETRIC_RECEIPT_SCHEMA = (
    "transcribe-audio.plan0063-biometric-private-copy-rehearsal-receipt.v1"
)
COMPLETE_RECEIPT_SCHEMA = (
    "transcribe-audio.plan0063-complete-private-copy-rehearsal-receipt.v1"
)
PROFILE_STATE_NAMES = ("profiles.sqlite3", "profiles", "authority")
REFERENCE_DATABASE_NAME = "references.sqlite3"
PROFILE_DATABASE_NAME = "profiles.sqlite3"
EXPECTED_PRODUCTION_ADAPTERS = {
    "speechbrain_ecapa_tdnn",
    "wespeaker_campplus",
    "wespeaker_resnet34",
}
REFERENCE_SOURCE_KEYS = (
    "reference_id",
    "source_blob_id",
    "source_sha256",
    "recording_id",
    "conversation_id",
    "speaker_label_id",
    "session_id",
    "start_seconds",
    "end_seconds",
    "source_duration_seconds",
    "quality_evidence",
    "lineage",
    "device_class",
    "acoustic_conditions",
)


class Plan0063BiometricRehearsalError(ValueError):
    """Raised when the biometric rehearsal cannot remain exact and private."""


def _fail(message: str) -> None:
    raise Plan0063BiometricRehearsalError(message)


def _content_hash(value: Mapping[str, Any], field: str) -> str:
    claimed = str(value.get("content_sha256") or "")
    core = {key: item for key, item in value.items() if key != "content_sha256"}
    if len(claimed) != 64 or canonical_artifact_hash(core) != claimed:
        _fail(f"The {field} content hash is invalid.")
    return claimed


def _paths(runtime_root: Path, transition_sha256: str) -> dict[str, Path]:
    common = canonical_rehearsal.rehearsal_paths(
        runtime_root, transition_sha256
    )
    biometric = common["run"] / "biometric"
    return {
        **common,
        "biometric": biometric,
        "reference_working": biometric / "reference-working",
        "reference_baseline": biometric / "reference-baseline",
        "profile_working": biometric / "profile-working",
        "profile_baseline": biometric / "profile-baseline",
        "biometric_manifest": biometric / "rehearsal.json",
        "biometric_receipt": biometric / "receipt.json",
        "complete_receipt": common["run"] / "complete-rehearsal-receipt.json",
    }


def _selected_children(root: Path, names: Sequence[str] | None) -> list[Path]:
    if names is None:
        return sorted(root.iterdir(), key=lambda item: item.name)
    result = []
    for name in names:
        selected = root / name
        if selected.exists() or selected.is_symlink():
            result.append(selected)
    return sorted(result, key=lambda item: item.name)


def _validate_tree_entry(path: Path) -> os.stat_result:
    if path.is_symlink():
        _fail("Private biometric state must not contain symlinks.")
    metadata = path.stat()
    if path.is_file():
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            _fail("Private biometric state must contain independent regular files.")
    elif not path.is_dir():
        _fail("Private biometric state contains an unsupported filesystem entry.")
    return metadata


def _tree_snapshot(
    root: Path, *, names: Sequence[str] | None = None
) -> dict[str, Any]:
    selected_root = root.expanduser().absolute()
    if (
        not selected_root.is_dir()
        or selected_root.is_symlink()
        or stat.S_IMODE(selected_root.stat().st_mode) != 0o700
    ):
        _fail("Private biometric state root must be a mode-0700 directory.")
    files = []
    directories = [{"path": ".", "mode": "0700"}]
    for child in _selected_children(selected_root, names):
        _validate_tree_entry(child)
        candidates = [child]
        if child.is_dir():
            candidates.extend(
                sorted(child.rglob("*"), key=lambda item: str(item.relative_to(selected_root)))
            )
        for path in candidates:
            metadata = _validate_tree_entry(path)
            relative = str(path.relative_to(selected_root))
            mode = stat.S_IMODE(metadata.st_mode)
            if path.is_dir():
                if mode != 0o700:
                    _fail("Private biometric directories must be mode 0700.")
                directories.append({"path": relative, "mode": "0700"})
                continue
            if mode != 0o600:
                _fail("Private biometric files must be mode 0600.")
            files.append(
                {
                    "path": relative,
                    "mode": "0600",
                    "bytes": metadata.st_size,
                    "sha256": sha256_file(path),
                }
            )
    core = {
        "directories": sorted(directories, key=lambda item: item["path"]),
        "files": sorted(files, key=lambda item: item["path"]),
    }
    return {**core, "snapshot_sha256": canonical_artifact_hash(core)}


def _secure_tree(root: Path) -> None:
    selected = root.expanduser().absolute()
    _validate_tree_entry(selected)
    selected.chmod(0o700)
    for path in sorted(selected.rglob("*"), key=str):
        _validate_tree_entry(path)
        path.chmod(0o700 if path.is_dir() else 0o600)


def _copy_state(
    source_root: Path,
    destination_root: Path,
    *,
    database_name: str,
    private_root: Path,
    names: Sequence[str] | None = None,
) -> None:
    source = source_root.expanduser().absolute()
    destination = destination_root.expanduser().absolute()
    database = source / database_name
    if (
        not source.is_dir()
        or source.is_symlink()
        or not database.is_file()
        or database.is_symlink()
    ):
        _fail("The live biometric state database is unavailable or unsafe.")
    _tree_snapshot(source, names=names)
    ensure_private_tree(private_root, destination)
    for child in _selected_children(source, names):
        if child.name in {database_name, f"{database_name}-wal", f"{database_name}-shm"}:
            continue
        _validate_tree_entry(child)
        target = destination / child.name
        if child.is_dir():
            shutil.copytree(child, target, copy_function=shutil.copy2)
        else:
            shutil.copy2(child, target)
    orchestration._sqlite_backup(database, destination / database_name)
    _secure_tree(destination)


def _copy_baseline(working: Path, baseline: Path) -> None:
    if baseline.exists() or baseline.is_symlink():
        _fail("A biometric baseline copy already exists.")
    shutil.copytree(working, baseline, copy_function=shutil.copy2)
    _secure_tree(baseline)


def _restore_baseline(working: Path, baseline: Path, private_root: Path) -> None:
    selected_working = working.expanduser().absolute()
    selected_baseline = baseline.expanduser().absolute()
    selected_private = private_root.expanduser().absolute()
    try:
        selected_working.relative_to(selected_private)
        selected_baseline.relative_to(selected_private)
    except ValueError as exc:
        raise Plan0063BiometricRehearsalError(
            "Biometric rollback paths escaped the private rehearsal root."
        ) from exc
    if (
        not selected_working.is_dir()
        or selected_working.is_symlink()
        or not selected_baseline.is_dir()
        or selected_baseline.is_symlink()
    ):
        _fail("Biometric rollback state is unavailable or unsafe.")
    shutil.rmtree(selected_working)
    shutil.copytree(selected_baseline, selected_working, copy_function=shutil.copy2)
    _secure_tree(selected_working)


def _store_snapshot(
    root: Path,
    *,
    database_name: str,
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    tree = _tree_snapshot(root, names=names)
    database = canonical_rehearsal._database_snapshot(
        root.expanduser().absolute() / database_name
    )
    core = {"tree": tree, "database": database}
    return {**core, "snapshot_sha256": canonical_artifact_hash(core)}


def _table_count(snapshot: Mapping[str, Any], table: str) -> int:
    database = snapshot.get("database")
    tables = database.get("tables") if isinstance(database, Mapping) else None
    row = tables.get(table) if isinstance(tables, Mapping) else None
    if not isinstance(row, Mapping):
        _fail(f"The biometric copy is missing table {table}.")
    return int(row.get("count") or 0)


def _reference_sources(unit: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_sources = unit.get("sources")
    if not isinstance(raw_sources, list):
        _fail("An enrollment unit has no reviewed source list.")
    if unit.get("status") == "ineligible_no_selected_source":
        if raw_sources or unit.get("source_count") != 0:
            _fail("An ineligible enrollment unit retained selected sources.")
        return []
    if unit.get("status") != "source_selected" or unit.get("source_count") != len(
        raw_sources
    ):
        _fail("An enrollment unit source status or count drifted.")
    sources = []
    for raw in raw_sources:
        if not isinstance(raw, Mapping):
            _fail("A reviewed biometric source is invalid.")
        if (
            raw.get("decision") != "include"
            or raw.get("future_holdout_excluded") is not True
            or raw.get("data_split") != "development_training_candidate"
        ):
            _fail("A biometric source is not reviewed development-only evidence.")
        source = {key: raw.get(key) for key in REFERENCE_SOURCE_KEYS}
        if any(source.get(key) is None for key in REFERENCE_SOURCE_KEYS):
            _fail("A reviewed biometric source lost governed reference fields.")
        sources.append(source)
    return sorted(sources, key=lambda item: str(item["reference_id"]))


def _enrollment_units(
    transition: Mapping[str, Any]
) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    raw_units = transition.get("enrollment_units")
    if not isinstance(raw_units, list) or len(raw_units) > 5:
        _fail("The reviewed biometric enrollment denominator is invalid.")
    result = []
    selected_source_count = 0
    seen_people: set[str] = set()
    for raw in raw_units:
        if not isinstance(raw, Mapping):
            _fail("A reviewed enrollment unit is invalid.")
        unit = dict(raw)
        person_id = str(unit.get("person_id") or "")
        if not person_id or person_id in seen_people:
            _fail("A reviewed enrollment person is missing or duplicated.")
        seen_people.add(person_id)
        sources = _reference_sources(unit)
        selected_source_count += len(sources)
        if sources:
            result.append((unit, sources))
    metrics = transition.get("metrics")
    if (
        not isinstance(metrics, Mapping)
        or metrics.get("enrollment_unit_count") != len(raw_units)
        or metrics.get("source_feasible_enrollment_unit_count") != len(result)
        or metrics.get("included_source_count") != selected_source_count
    ):
        _fail("The reviewed biometric transition metrics drifted.")
    return sorted(result, key=lambda item: str(item[0]["person_id"]))


def _approval(
    *,
    action: str,
    profile_id: str,
    person_ref_id: str,
    source_set_sha256: str | None,
    expected_generation_id: str | None,
    transition: Mapping[str, Any],
) -> dict[str, Any]:
    identity = {
        "action": action,
        "profile_id": profile_id,
        "person_ref_id": person_ref_id,
        "source_set_sha256": source_set_sha256,
        "expected_generation_id": expected_generation_id,
        "transition_sha256": transition["content_sha256"],
    }
    return {
        "schema_version": references.APPROVAL_SCHEMA,
        "approval_id": "plan0063-private-approval-"
        + canonical_artifact_hash(identity)[:24],
        "reviewer_ref_id": "plan0063-private-rehearsal",
        "reviewed_at": transition["reviewed_at"],
        "purpose": f"biometric_reference_{action}",
        "scope": {
            "profile_id": profile_id,
            "person_ref_id": person_ref_id,
            "source_set_sha256": source_set_sha256,
            "expected_generation_id": expected_generation_id,
        },
    }


def _reference_profile_id(
    transition_sha256: str, person_ref_id: str
) -> str:
    identity = {
        "transition_sha256": transition_sha256,
        "person_ref_id": person_ref_id,
    }
    return "plan0063-reference-" + canonical_artifact_hash(identity)[:24]


def _p1_windows(
    sources: Sequence[Mapping[str, Any]],
    *,
    lineage_cache: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    windows = []
    for source in sources:
        lineage = source.get("lineage")
        if (
            not isinstance(lineage, Mapping)
            or lineage.get("authority") != "p1_audio_derivative_replay"
        ):
            _fail("Plan 0063 profile materialization requires exact P1 lineage.")
        run_id = str(lineage.get("run_id") or "")
        if run_id not in lineage_cache:
            try:
                metadata = resolve_derivative_lineage_receipt(
                    run_id,
                    replay_receipt_sha256=str(
                        lineage.get("replay_receipt_sha256") or ""
                    ),
                    runtime_root=Path(str(lineage.get("runtime_root") or "")),
                )
                active = resolve_active_derivative(
                    run_id,
                    runtime_root=Path(str(lineage.get("runtime_root") or "")),
                )
            except AudioDerivativeError as exc:
                raise Plan0063BiometricRehearsalError(
                    "A reviewed P1 lineage is not replay-valid."
                ) from exc
            if dict(lineage) != metadata:
                _fail("Stored Plan 0063 P1 lineage drifted.")
            lineage_cache[run_id] = active
        active = lineage_cache[run_id]
        if (
            active.get("source_blob_id") != source.get("source_blob_id")
            or active.get("source_sha256") != source.get("source_sha256")
            or active.get("artifact_sha256") != lineage.get("artifact_sha256")
            or active.get("audio_quality_sha256")
            != source.get("quality_evidence", {}).get("sha256")
            or float(active.get("derived_audio", {}).get("source_duration_seconds"))
            != float(source.get("source_duration_seconds"))
        ):
            _fail("The reviewed profile window lost its exact P1 source binding.")
        audio_path = Path(str(active.get("artifact_path") or ""))
        require_private_file(audio_path, audio_path.parent)
        try:
            with wave.open(str(audio_path), "rb") as reader:
                if (
                    reader.getnchannels() != 1
                    or reader.getsampwidth() != 2
                    or reader.getframerate() != 16_000
                    or reader.getcomptype() != "NONE"
                ):
                    _fail("The reviewed P1 derivative is not canonical PCM.")
                start_frame = round(float(source["start_seconds"]) * 16_000)
                end_frame = round(float(source["end_seconds"]) * 16_000)
                if (
                    start_frame < 0
                    or end_frame <= start_frame
                    or end_frame > reader.getnframes()
                ):
                    _fail("A reviewed profile window is outside its P1 derivative.")
                reader.setpos(start_frame)
                payload = reader.readframes(end_frame - start_frame)
        except (EOFError, OSError, TypeError, ValueError, wave.Error) as exc:
            raise Plan0063BiometricRehearsalError(
                "A reviewed P1 profile window is unreadable."
            ) from exc
        sample_count = end_frame - start_frame
        if len(payload) != sample_count * 2:
            _fail("A reviewed P1 profile window is truncated.")
        samples = tuple(
            value / 32768.0
            for value in struct.unpack(f"<{sample_count}h", payload)
        )
        windows.append(
            {"session_id": str(source["session_id"]), "samples": samples}
        )
    return windows


def _adapters(
    supplied: Mapping[str, verification.VerificationAdapter] | None,
    *,
    test_mode: bool,
) -> dict[str, verification.VerificationAdapter]:
    if supplied is not None and not test_mode:
        _fail("Custom biometric adapters are limited to deterministic tests.")
    selected = dict(supplied or verification.adapter_registry())
    if not selected or any(
        key != adapter.candidate_id for key, adapter in selected.items()
    ):
        _fail("The biometric adapter inventory is invalid.")
    if not test_mode and set(selected) != EXPECTED_PRODUCTION_ADAPTERS:
        _fail("The production biometric adapter inventory drifted.")
    return selected


def _lifecycle_reference_action(
    action: str,
    *,
    profile_id: str,
    person_ref_id: str,
    transition: Mapping[str, Any],
    runtime_root: Path,
) -> dict[str, Any]:
    current = references.replay_reference(profile_id, runtime_root=runtime_root)
    approval = _approval(
        action=action,
        profile_id=profile_id,
        person_ref_id=person_ref_id,
        source_set_sha256=None,
        expected_generation_id=str(current["head_generation_id"]),
        transition=transition,
    )
    plan = references.dry_run(
        action,
        profile_id=profile_id,
        approval=approval,
        runtime_root=runtime_root,
    )
    return references.apply_change(
        str(plan["run_id"]),
        approval_token=str(plan["required_approval_token"]),
        runtime_root=runtime_root,
    )


def _database_delta(
    before: Mapping[str, Any], after: Mapping[str, Any], table: str
) -> int:
    return _table_count(after, table) - _table_count(before, table)


def replay_biometric_rehearsal(
    *,
    transition_sha256: str,
    live_reference_root: Path,
    live_profile_root: Path,
    runtime_root: Path = canonical_rehearsal.DEFAULT_RUNTIME_ROOT,
) -> dict[str, Any]:
    paths = _paths(runtime_root, transition_sha256)
    for key in ("transition", "biometric_manifest", "biometric_receipt"):
        require_private_file(paths[key], paths["root"])
    transition = read_private_object(paths["transition"])
    canonical_rehearsal.validate_reviewed_transition(transition)
    if transition["content_sha256"] != transition_sha256:
        _fail("The selected transition does not match the biometric rehearsal.")
    manifest = read_private_object(paths["biometric_manifest"])
    receipt = read_private_object(paths["biometric_receipt"])
    _content_hash(manifest, "biometric rehearsal manifest")
    reference_baseline = _store_snapshot(
        paths["reference_baseline"], database_name=REFERENCE_DATABASE_NAME
    )
    reference_working = _store_snapshot(
        paths["reference_working"], database_name=REFERENCE_DATABASE_NAME
    )
    profile_baseline = _store_snapshot(
        paths["profile_baseline"], database_name=PROFILE_DATABASE_NAME
    )
    profile_working = _store_snapshot(
        paths["profile_working"], database_name=PROFILE_DATABASE_NAME
    )
    live_reference = _store_snapshot(
        live_reference_root, database_name=REFERENCE_DATABASE_NAME
    )
    live_profile = _store_snapshot(
        live_profile_root,
        database_name=PROFILE_DATABASE_NAME,
        names=PROFILE_STATE_NAMES,
    )
    receipt_core = {
        key: value for key, value in receipt.items() if key != "content_sha256"
    }
    if (
        manifest.get("schema_version") != BIOMETRIC_REHEARSAL_SCHEMA
        or manifest.get("transition_sha256") != transition_sha256
        or manifest.get("reference_baseline_snapshot") != reference_baseline
        or manifest.get("reference_rolled_back_snapshot") != reference_working
        or manifest.get("profile_baseline_snapshot") != profile_baseline
        or manifest.get("profile_rolled_back_snapshot") != profile_working
        or reference_working != reference_baseline
        or profile_working != profile_baseline
        or manifest.get("live_reference_snapshot_before") != live_reference
        or manifest.get("live_reference_snapshot_after") != live_reference
        or manifest.get("live_profile_snapshot_before") != live_profile
        or manifest.get("live_profile_snapshot_after") != live_profile
        or receipt.get("schema_version") != BIOMETRIC_RECEIPT_SCHEMA
        or receipt.get("manifest_sha256")
        != sha256_file(paths["biometric_manifest"])
        or receipt.get("transition_file_sha256")
        != sha256_file(paths["transition"])
        or receipt.get("content_sha256") != canonical_artifact_hash(receipt_core)
        or receipt.get("live_mutation_count") != 0
    ):
        _fail("The biometric private-copy rehearsal replay drifted.")
    return {
        **receipt,
        "manifest_path": str(paths["biometric_manifest"]),
        "receipt_path": str(paths["biometric_receipt"]),
        "transition_path": str(paths["transition"]),
        "idempotent_replay": True,
    }


def apply_reviewed_biometric_transition(
    transition: Mapping[str, Any],
    *,
    reference_root: Path,
    profile_root: Path,
    reference_baseline: Mapping[str, Any],
    profile_baseline: Mapping[str, Any],
    adapters: Mapping[str, verification.VerificationAdapter] | None = None,
    test_mode: bool = False,
) -> dict[str, Any]:
    """Apply one reviewed Plan 0063 transition to governed biometric stores."""

    transition_sha256 = canonical_rehearsal.validate_reviewed_transition(
        transition
    )
    selected_adapters = _adapters(adapters, test_mode=test_mode)
    units = _enrollment_units(transition)
    created_references = []
    created_profiles = []
    lineage_cache: dict[str, dict[str, Any]] = {}
    for unit, sources in units:
        person_ref_id = str(unit["person_id"])
        profile_id = _reference_profile_id(transition_sha256, person_ref_id)
        source_hash = references.source_set_sha256(sources)
        approval = _approval(
            action="create",
            profile_id=profile_id,
            person_ref_id=person_ref_id,
            source_set_sha256=source_hash,
            expected_generation_id=None,
            transition=transition,
        )
        plan = references.dry_run(
            "create",
            profile_id=profile_id,
            person_ref_id=person_ref_id,
            sources=sources,
            approval=approval,
            runtime_root=reference_root,
        )
        applied = references.apply_change(
            str(plan["run_id"]),
            approval_token=str(plan["required_approval_token"]),
            sources=sources,
            runtime_root=reference_root,
        )
        replayed = references.replay_reference(
            profile_id, runtime_root=reference_root
        )
        resolved = references.resolve_eligible_reference(
            person_ref_id, runtime_root=reference_root
        )
        if (
            replayed.get("lifecycle_state") != "verified_active"
            or replayed.get("eligible_for_materialization") is not True
            or resolved.get("profile_id") != profile_id
        ):
            _fail("A biometric reference did not become eligible.")
        created_references.append(
            {
                "person_ref_id": person_ref_id,
                "profile_id": profile_id,
                "generation_id": resolved["generation_id"],
                "generation_sha256": resolved["generation_sha256"],
                "source_set_sha256": source_hash,
                "source_count": len(sources),
                "apply_receipt_sha256": canonical_artifact_hash(
                    {
                        key: value
                        for key, value in applied.items()
                        if key not in {"receipt_anchor_path", "idempotent_replay"}
                    }
                ),
            }
        )
        windows = _p1_windows(sources, lineage_cache=lineage_cache)
        for candidate_id in sorted(selected_adapters):
            profile = verification._materialize_profile_core(
                resolved=resolved,
                adapter=selected_adapters[candidate_id],
                windows=windows,
                preprocessing={
                    "method_id": "no_enhancement",
                    "revision": transition_sha256,
                },
                runtime_root=profile_root,
                p3_runtime_root=reference_root,
            )
            replayed_profile = verification.replay_profile(
                str(profile["profile_id"]), runtime_root=profile_root
            )
            if (
                replayed_profile.get("lifecycle_state") != "active"
                or replayed_profile.get("private_bytes_present") is not True
                or replayed_profile.get("person_ref_id") != person_ref_id
            ):
                _fail("A biometric model profile did not become active.")
            created_profiles.append(dict(profile))

    reference_snapshot = _store_snapshot(
        reference_root, database_name=REFERENCE_DATABASE_NAME
    )
    profile_snapshot = _store_snapshot(
        profile_root,
        database_name=PROFILE_DATABASE_NAME,
        names=PROFILE_STATE_NAMES,
    )
    expected_reference_count = len(units)
    expected_profile_count = len(units) * len(selected_adapters)
    expected_source_count = sum(len(sources) for _, sources in units)
    expected_reference_deltas = {
        "profiles": expected_reference_count,
        "generations": expected_reference_count,
        "person_heads": expected_reference_count,
        "source_claims": expected_source_count,
        "descendants": expected_profile_count,
    }
    if any(
        _database_delta(reference_baseline, reference_snapshot, table) != count
        for table, count in expected_reference_deltas.items()
    ) or _database_delta(
        profile_baseline, profile_snapshot, "profiles"
    ) != expected_profile_count:
        _fail("The biometric apply counts did not reconcile.")
    adapter_inventory = [
        {
            "candidate_id": candidate_id,
            "revision_sha": selected_adapters[candidate_id].revision_sha,
            "embedding_dimension": selected_adapters[
                candidate_id
            ].embedding_dimension,
        }
        for candidate_id in sorted(selected_adapters)
    ]
    return {
        "reference_snapshot": reference_snapshot,
        "profile_snapshot": profile_snapshot,
        "created_references": created_references,
        "created_profiles": created_profiles,
        "reference_count": expected_reference_count,
        "profile_count": expected_profile_count,
        "source_count": expected_source_count,
        "adapter_count": len(selected_adapters),
        "adapter_inventory": adapter_inventory,
        "reference_database_deltas": expected_reference_deltas,
    }


def rehearse_biometric_copy(
    transition: Mapping[str, Any],
    *,
    live_reference_root: Path,
    live_profile_root: Path,
    runtime_root: Path = canonical_rehearsal.DEFAULT_RUNTIME_ROOT,
    adapters: Mapping[str, verification.VerificationAdapter] | None = None,
    test_mode: bool = False,
) -> dict[str, Any]:
    """Apply, lifecycle-delete, and exactly restore biometric state copies."""

    transition_sha256 = canonical_rehearsal.validate_reviewed_transition(
        transition
    )
    _adapters(adapters, test_mode=test_mode)
    paths = _paths(runtime_root, transition_sha256)
    if paths["biometric_receipt"].exists():
        return replay_biometric_rehearsal(
            transition_sha256=transition_sha256,
            live_reference_root=live_reference_root,
            live_profile_root=live_profile_root,
            runtime_root=runtime_root,
        )
    if paths["biometric"].exists():
        _fail("A partial biometric private-copy rehearsal already exists.")
    ensure_private_tree(paths["root"], paths["run"])
    ensure_private_tree(paths["root"], paths["biometric"])
    live_reference_before = _store_snapshot(
        live_reference_root, database_name=REFERENCE_DATABASE_NAME
    )
    live_profile_before = _store_snapshot(
        live_profile_root,
        database_name=PROFILE_DATABASE_NAME,
        names=PROFILE_STATE_NAMES,
    )
    try:
        _copy_state(
            live_reference_root,
            paths["reference_working"],
            database_name=REFERENCE_DATABASE_NAME,
            private_root=paths["root"],
        )
        _copy_baseline(paths["reference_working"], paths["reference_baseline"])
        _copy_state(
            live_profile_root,
            paths["profile_working"],
            database_name=PROFILE_DATABASE_NAME,
            private_root=paths["root"],
            names=PROFILE_STATE_NAMES,
        )
        _copy_baseline(paths["profile_working"], paths["profile_baseline"])
        reference_baseline = _store_snapshot(
            paths["reference_baseline"], database_name=REFERENCE_DATABASE_NAME
        )
        profile_baseline = _store_snapshot(
            paths["profile_baseline"], database_name=PROFILE_DATABASE_NAME
        )

        apply_result = apply_reviewed_biometric_transition(
            transition,
            reference_root=paths["reference_working"],
            profile_root=paths["profile_working"],
            reference_baseline=reference_baseline,
            profile_baseline=profile_baseline,
            adapters=adapters,
            test_mode=test_mode,
        )
        created_references = apply_result["created_references"]
        created_profiles = apply_result["created_profiles"]
        reference_applied = apply_result["reference_snapshot"]
        profile_applied = apply_result["profile_snapshot"]
        expected_reference_count = apply_result["reference_count"]
        expected_profile_count = apply_result["profile_count"]
        expected_source_count = apply_result["source_count"]
        expected_reference_deltas = apply_result["reference_database_deltas"]

        deleted_profiles = []
        for profile in reversed(created_profiles):
            profile_id = str(profile["profile_id"])
            verification.withdraw_profile(
                profile_id,
                reason="plan0063_private_rollback",
                runtime_root=paths["profile_working"],
                p3_runtime_root=paths["reference_working"],
            )
            deleted = verification.delete_profile(
                profile_id,
                reason="plan0063_private_rollback",
                runtime_root=paths["profile_working"],
                p3_runtime_root=paths["reference_working"],
            )
            if (
                deleted.get("lifecycle_state") != "deleted"
                or deleted.get("private_bytes_present") is not False
            ):
                _fail("A private biometric model profile did not delete cleanly.")
            deleted_profiles.append(
                {
                    "profile_id": profile_id,
                    "lifecycle_state": deleted["lifecycle_state"],
                    "private_bytes_present": deleted["private_bytes_present"],
                }
            )

        deleted_references = []
        for reference in reversed(created_references):
            profile_id = str(reference["profile_id"])
            person_ref_id = str(reference["person_ref_id"])
            _lifecycle_reference_action(
                "withdraw",
                profile_id=profile_id,
                person_ref_id=person_ref_id,
                transition=transition,
                runtime_root=paths["reference_working"],
            )
            _lifecycle_reference_action(
                "delete",
                profile_id=profile_id,
                person_ref_id=person_ref_id,
                transition=transition,
                runtime_root=paths["reference_working"],
            )
            deleted = references.replay_reference(
                profile_id, runtime_root=paths["reference_working"]
            )
            if (
                deleted.get("lifecycle_state") != "verified_deleted"
                or deleted.get("eligible_for_materialization") is not False
            ):
                _fail("A private biometric reference did not delete cleanly.")
            deleted_references.append(
                {
                    "profile_id": profile_id,
                    "lifecycle_state": deleted["lifecycle_state"],
                    "eligible_for_materialization": deleted[
                        "eligible_for_materialization"
                    ],
                }
            )

        reference_logical_rollback = _store_snapshot(
            paths["reference_working"], database_name=REFERENCE_DATABASE_NAME
        )
        profile_logical_rollback = _store_snapshot(
            paths["profile_working"], database_name=PROFILE_DATABASE_NAME
        )
        _restore_baseline(
            paths["reference_working"],
            paths["reference_baseline"],
            paths["root"],
        )
        _restore_baseline(
            paths["profile_working"],
            paths["profile_baseline"],
            paths["root"],
        )
        reference_rolled_back = _store_snapshot(
            paths["reference_working"], database_name=REFERENCE_DATABASE_NAME
        )
        profile_rolled_back = _store_snapshot(
            paths["profile_working"], database_name=PROFILE_DATABASE_NAME
        )
        if (
            reference_rolled_back != reference_baseline
            or profile_rolled_back != profile_baseline
        ):
            _fail("The exact biometric baseline was not restored.")
        live_reference_after = _store_snapshot(
            live_reference_root, database_name=REFERENCE_DATABASE_NAME
        )
        live_profile_after = _store_snapshot(
            live_profile_root,
            database_name=PROFILE_DATABASE_NAME,
            names=PROFILE_STATE_NAMES,
        )
        if (
            live_reference_after != live_reference_before
            or live_profile_after != live_profile_before
        ):
            _fail("Live biometric state changed during the private rehearsal.")

        write_immutable_private_json(paths["transition"], dict(transition))
        adapter_inventory = apply_result["adapter_inventory"]
        manifest_core = {
            "schema_version": BIOMETRIC_REHEARSAL_SCHEMA,
            "status": "biometric_private_apply_lifecycle_rollback_and_restore_proved",
            "transition_sha256": transition_sha256,
            "adapter_inventory": adapter_inventory,
            "reviewed_enrollment_unit_count": len(
                transition.get("enrollment_units") or []
            ),
            "applied_reference_count": expected_reference_count,
            "applied_profile_count": expected_profile_count,
            "applied_source_count": expected_source_count,
            "reference_database_deltas": expected_reference_deltas,
            "profile_database_deltas": {"profiles": expected_profile_count},
            "created_references": created_references,
            "created_profiles": created_profiles,
            "deleted_references": deleted_references,
            "deleted_profiles": deleted_profiles,
            "live_reference_snapshot_before": live_reference_before,
            "reference_baseline_snapshot": reference_baseline,
            "reference_applied_snapshot": reference_applied,
            "reference_logical_rollback_snapshot": reference_logical_rollback,
            "reference_rolled_back_snapshot": reference_rolled_back,
            "live_reference_snapshot_after": live_reference_after,
            "live_profile_snapshot_before": live_profile_before,
            "profile_baseline_snapshot": profile_baseline,
            "profile_applied_snapshot": profile_applied,
            "profile_logical_rollback_snapshot": profile_logical_rollback,
            "profile_rolled_back_snapshot": profile_rolled_back,
            "live_profile_snapshot_after": live_profile_after,
            "copy_apply_count": 1,
            "copy_rollback_count": 1,
            "a1_authorized": False,
            "live_mutation_count": 0,
        }
        manifest = {
            **manifest_core,
            "content_sha256": canonical_artifact_hash(manifest_core),
        }
        write_immutable_private_json(paths["biometric_manifest"], manifest)
        receipt_core = {
            "schema_version": BIOMETRIC_RECEIPT_SCHEMA,
            "status": "biometric_private_apply_lifecycle_rollback_and_restore_proved",
            "transition_sha256": transition_sha256,
            "transition_file_sha256": sha256_file(paths["transition"]),
            "manifest_sha256": sha256_file(paths["biometric_manifest"]),
            "reference_baseline_sha256": reference_baseline["snapshot_sha256"],
            "reference_rolled_back_sha256": reference_rolled_back[
                "snapshot_sha256"
            ],
            "profile_baseline_sha256": profile_baseline["snapshot_sha256"],
            "profile_rolled_back_sha256": profile_rolled_back[
                "snapshot_sha256"
            ],
            "applied_reference_count": expected_reference_count,
            "applied_profile_count": expected_profile_count,
            "applied_source_count": expected_source_count,
            "copy_apply_count": 1,
            "copy_rollback_count": 1,
            "biometric_rehearsal_complete": True,
            "a1_authorized": False,
            "live_mutation_count": 0,
        }
        receipt = {
            **receipt_core,
            "content_sha256": canonical_artifact_hash(receipt_core),
        }
        write_immutable_private_json(paths["biometric_receipt"], receipt)
        return {
            **receipt,
            "manifest_path": str(paths["biometric_manifest"]),
            "receipt_path": str(paths["biometric_receipt"]),
            "transition_path": str(paths["transition"]),
            "idempotent_replay": False,
        }
    except Exception:
        if paths["biometric"].exists() and not paths["biometric_receipt"].exists():
            shutil.rmtree(paths["biometric"])
        raise


def replay_complete_private_rehearsal(
    *,
    transition_sha256: str,
    live_store_root: Path,
    live_reference_root: Path,
    live_profile_root: Path,
    runtime_root: Path = canonical_rehearsal.DEFAULT_RUNTIME_ROOT,
) -> dict[str, Any]:
    """Replay the joined knowledge and biometric private-copy proof."""

    paths = _paths(runtime_root, transition_sha256)
    require_private_file(paths["complete_receipt"], paths["root"])
    receipt = read_private_object(paths["complete_receipt"])
    knowledge = canonical_rehearsal.replay_knowledge_rehearsal(
        transition_sha256=transition_sha256,
        live_store_root=live_store_root,
        runtime_root=runtime_root,
    )
    biometric = replay_biometric_rehearsal(
        transition_sha256=transition_sha256,
        live_reference_root=live_reference_root,
        live_profile_root=live_profile_root,
        runtime_root=runtime_root,
    )
    receipt_core = {
        key: value for key, value in receipt.items() if key != "content_sha256"
    }
    if (
        receipt.get("schema_version") != COMPLETE_RECEIPT_SCHEMA
        or receipt.get("transition_sha256") != transition_sha256
        or receipt.get("knowledge_receipt_content_sha256")
        != knowledge.get("content_sha256")
        or receipt.get("knowledge_receipt_file_sha256")
        != sha256_file(Path(str(knowledge["receipt_path"])))
        or receipt.get("biometric_receipt_content_sha256")
        != biometric.get("content_sha256")
        or receipt.get("biometric_receipt_file_sha256")
        != sha256_file(Path(str(biometric["receipt_path"])))
        or receipt.get("logical_transition_apply_count") != 1
        or receipt.get("logical_transition_rollback_count") != 1
        or receipt.get("a1_request_ready")
        is not (receipt.get("test_mode") is False)
        or receipt.get("a1_authorized") is not False
        or receipt.get("live_mutation_count") != 0
        or receipt.get("content_sha256") != canonical_artifact_hash(receipt_core)
    ):
        _fail("The complete private-copy rehearsal replay drifted.")
    return {
        **receipt,
        "receipt_path": str(paths["complete_receipt"]),
        "knowledge_receipt_path": knowledge["receipt_path"],
        "biometric_receipt_path": biometric["receipt_path"],
        "idempotent_replay": True,
    }


def rehearse_complete_private_copy(
    transition: Mapping[str, Any],
    *,
    live_store_root: Path,
    live_reference_root: Path,
    live_profile_root: Path,
    runtime_root: Path = canonical_rehearsal.DEFAULT_RUNTIME_ROOT,
    adapters: Mapping[str, verification.VerificationAdapter] | None = None,
    test_mode: bool = False,
) -> dict[str, Any]:
    """Run one logical transition rehearsal across all three governed stores."""

    transition_sha256 = canonical_rehearsal.validate_reviewed_transition(
        transition
    )
    paths = _paths(runtime_root, transition_sha256)
    if paths["complete_receipt"].exists():
        return replay_complete_private_rehearsal(
            transition_sha256=transition_sha256,
            live_store_root=live_store_root,
            live_reference_root=live_reference_root,
            live_profile_root=live_profile_root,
            runtime_root=runtime_root,
        )
    knowledge = canonical_rehearsal.rehearse_knowledge_copy(
        transition,
        live_store_root=live_store_root,
        runtime_root=runtime_root,
    )
    biometric = rehearse_biometric_copy(
        transition,
        live_reference_root=live_reference_root,
        live_profile_root=live_profile_root,
        runtime_root=runtime_root,
        adapters=adapters,
        test_mode=test_mode,
    )
    receipt_core = {
        "schema_version": COMPLETE_RECEIPT_SCHEMA,
        "status": "complete_private_apply_and_rollback_proved",
        "transition_sha256": transition_sha256,
        "knowledge_receipt_content_sha256": knowledge["content_sha256"],
        "knowledge_receipt_file_sha256": sha256_file(
            Path(str(knowledge["receipt_path"]))
        ),
        "biometric_receipt_content_sha256": biometric["content_sha256"],
        "biometric_receipt_file_sha256": sha256_file(
            Path(str(biometric["receipt_path"]))
        ),
        "applied_person_count": int(
            transition.get("metrics", {}).get("canonical_person_count") or 0
        ),
        "applied_reference_count": biometric["applied_reference_count"],
        "applied_profile_count": biometric["applied_profile_count"],
        "applied_source_count": biometric["applied_source_count"],
        "subsystem_copy_apply_counts": {"knowledge": 1, "biometric": 1},
        "subsystem_copy_rollback_counts": {"knowledge": 1, "biometric": 1},
        "logical_transition_apply_count": 1,
        "logical_transition_rollback_count": 1,
        "test_mode": test_mode,
        "a1_request_ready": not test_mode,
        "a1_authorized": False,
        "live_mutation_count": 0,
    }
    receipt = {
        **receipt_core,
        "content_sha256": canonical_artifact_hash(receipt_core),
    }
    write_immutable_private_json(paths["complete_receipt"], receipt)
    return {
        **receipt,
        "receipt_path": str(paths["complete_receipt"]),
        "knowledge_receipt_path": knowledge["receipt_path"],
        "biometric_receipt_path": biometric["receipt_path"],
        "idempotent_replay": False,
    }


__all__ = [
    "BIOMETRIC_RECEIPT_SCHEMA",
    "BIOMETRIC_REHEARSAL_SCHEMA",
    "COMPLETE_RECEIPT_SCHEMA",
    "Plan0063BiometricRehearsalError",
    "apply_reviewed_biometric_transition",
    "rehearse_biometric_copy",
    "rehearse_complete_private_copy",
    "replay_biometric_rehearsal",
    "replay_complete_private_rehearsal",
]
