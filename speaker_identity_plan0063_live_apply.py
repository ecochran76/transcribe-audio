"""A1-gated one-shot live apply for the joined Plan 0063 transition."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import acoustic_biometric_references as references
import acoustic_verification as verification
import speaker_identity_plan0063_a1_authority as a1
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


LIVE_APPLY_PREPARED_SCHEMA = "transcribe-audio.plan0063-live-apply-prepared.v1"
LIVE_APPLY_RECEIPT_SCHEMA = "transcribe-audio.plan0063-live-apply-receipt.v1"
DEFAULT_RUNTIME_ROOT = canonical_rehearsal.DEFAULT_RUNTIME_ROOT
SERVICES = ("transcripts.service", "transcribe-watch.service")
SHA256_RE = re.compile(r"[a-f0-9]{64}")


class Plan0063LiveApplyError(RuntimeError):
    """Raised when the exact A1-gated live transition cannot apply safely."""


class ServiceController(Protocol):
    def snapshot(self) -> dict[str, Any]: ...

    def quiesce(self) -> dict[str, Any]: ...

    def restore(self) -> dict[str, Any]: ...


def _fail(message: str) -> None:
    raise Plan0063LiveApplyError(message)


class SystemdServiceController:
    """Quiesce and restore only the two exact user-scoped transcript services."""

    def _run(self, arguments: Sequence[str]) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            ["systemctl", "--user", *arguments],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            _fail("Transcript service control failed.")
        return result

    def snapshot(self) -> dict[str, Any]:
        result = self._run(
            [
                "show",
                *SERVICES,
                "--property=Id,ActiveState,SubState,NRestarts",
                "--no-pager",
            ]
        )
        rows: dict[str, dict[str, Any]] = {}
        current: dict[str, str] = {}
        for line in result.stdout.splitlines() + [""]:
            if line.strip() and "=" in line:
                key, value = line.split("=", 1)
                current[key] = value
                continue
            if current:
                service_id = current.get("Id", "")
                if service_id:
                    rows[service_id] = {
                        "active_state": current.get("ActiveState"),
                        "sub_state": current.get("SubState"),
                        "nrestarts": int(current.get("NRestarts") or 0),
                    }
                current = {}
        if set(rows) != set(SERVICES):
            _fail("Transcript service state is incomplete.")
        return {"services": rows}

    def quiesce(self) -> dict[str, Any]:
        before = self.snapshot()
        if not all(
            row.get("active_state") in {"inactive", "failed"}
            for row in before["services"].values()
        ):
            self._run(["stop", *SERVICES])
        after = self.snapshot()
        if any(
            row.get("active_state") not in {"inactive", "failed"}
            for row in after["services"].values()
        ):
            _fail("Transcript services did not quiesce.")
        return after

    def restore(self) -> dict[str, Any]:
        self._run(["start", *SERVICES])
        after = self.snapshot()
        if any(
            row.get("active_state") != "active" or row.get("sub_state") != "running"
            for row in after["services"].values()
        ):
            _fail("Transcript services did not return to active/running.")
        return after


def _paths(runtime_root: Path, transition_sha256: str) -> dict[str, Path]:
    if not SHA256_RE.fullmatch(str(transition_sha256)):
        _fail("The reviewed transition hash is invalid.")
    root = runtime_root.expanduser().absolute()
    run = root / f"live-apply-{transition_sha256[:20]}"
    backup = run / "backup"
    return {
        "root": root,
        "run": run,
        "backup": backup,
        "knowledge_backup_root": backup / "knowledge",
        "knowledge_backup_db": backup / "knowledge" / transcript_store.DEFAULT_DB_NAME,
        "reference_backup": backup / "references",
        "profile_backup": backup / "profiles",
        "prepared": run / "prepared.json",
        "receipt": run / "receipt.json",
    }


def _load_transition(runtime_root: Path, transition_sha256: str) -> dict[str, Any]:
    path = canonical_rehearsal.rehearsal_paths(
        runtime_root, transition_sha256
    )["transition"]
    require_private_file(path, runtime_root.expanduser().absolute())
    transition = read_private_object(path)
    if canonical_rehearsal.validate_reviewed_transition(transition) != transition_sha256:
        _fail("The reviewed live transition replay drifted.")
    return transition


def _live_snapshots(
    *,
    live_store_root: Path,
    live_reference_root: Path,
    live_profile_root: Path,
) -> dict[str, Any]:
    return {
        "knowledge": canonical_rehearsal._database_snapshot(
            transcript_store.db_path(live_store_root)
        ),
        "references": biometric_rehearsal._store_snapshot(
            live_reference_root,
            database_name=biometric_rehearsal.REFERENCE_DATABASE_NAME,
        ),
        "profiles": biometric_rehearsal._store_snapshot(
            live_profile_root,
            database_name=biometric_rehearsal.PROFILE_DATABASE_NAME,
            names=biometric_rehearsal.PROFILE_STATE_NAMES,
        ),
    }


def _snapshot_hashes(snapshots: Mapping[str, Any]) -> dict[str, str]:
    return {
        key: canonical_artifact_hash(value)
        for key, value in sorted(snapshots.items())
    }


def _prepare_backups(
    paths: Mapping[str, Path],
    *,
    live_store_root: Path,
    live_reference_root: Path,
    live_profile_root: Path,
) -> dict[str, Any]:
    ensure_private_tree(paths["root"], paths["knowledge_backup_root"])
    shutil.copy2(
        transcript_store.db_path(live_store_root), paths["knowledge_backup_db"]
    )
    paths["knowledge_backup_db"].chmod(0o600)
    _copy_quiesced_store(
        live_reference_root,
        paths["reference_backup"],
        private_root=paths["root"],
    )
    _copy_quiesced_store(
        live_profile_root,
        paths["profile_backup"],
        private_root=paths["root"],
        names=biometric_rehearsal.PROFILE_STATE_NAMES,
    )
    snapshots = {
        "knowledge": canonical_rehearsal._database_snapshot(
            paths["knowledge_backup_db"]
        ),
        "references": biometric_rehearsal._store_snapshot(
            paths["reference_backup"],
            database_name=biometric_rehearsal.REFERENCE_DATABASE_NAME,
        ),
        "profiles": biometric_rehearsal._store_snapshot(
            paths["profile_backup"],
            database_name=biometric_rehearsal.PROFILE_DATABASE_NAME,
            names=biometric_rehearsal.PROFILE_STATE_NAMES,
        ),
    }
    return {"snapshots": snapshots, "snapshot_sha256s": _snapshot_hashes(snapshots)}


def _copy_quiesced_store(
    source_root: Path,
    destination_root: Path,
    *,
    private_root: Path,
    names: Sequence[str] | None = None,
) -> None:
    """Copy exact bytes after the owning services have been quiesced."""

    source = source_root.expanduser().absolute()
    destination = destination_root.expanduser().absolute()
    if not source.is_dir() or source.is_symlink():
        _fail("A quiesced live state root is unavailable or unsafe.")
    biometric_rehearsal._tree_snapshot(source, names=names)
    ensure_private_tree(private_root, destination)
    for child in biometric_rehearsal._selected_children(source, names):
        biometric_rehearsal._validate_tree_entry(child)
        target = destination / child.name
        if child.is_dir():
            shutil.copytree(child, target, copy_function=shutil.copy2)
        else:
            shutil.copy2(child, target)
    biometric_rehearsal._secure_tree(destination)


def _restore_database(source: Path, destination: Path) -> None:
    if not source.is_file() or source.is_symlink():
        _fail("The knowledge rollback backup is unavailable.")
    for suffix in ("-wal", "-shm"):
        companion = Path(str(destination) + suffix)
        if companion.exists():
            if companion.is_symlink() or not companion.is_file():
                _fail("A knowledge rollback companion is unsafe.")
            companion.unlink()
    descriptor, stage_name = tempfile.mkstemp(
        prefix=".plan0063-restore-", dir=destination.parent
    )
    os.close(descriptor)
    stage = Path(stage_name)
    try:
        shutil.copy2(source, stage)
        stage.chmod(0o600)
        os.replace(stage, destination)
    finally:
        if stage.exists():
            stage.unlink()


def _remove_entry(path: Path) -> None:
    if path.is_symlink():
        _fail("A live rollback target became a symlink.")
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        if not path.is_file():
            _fail("A live rollback target is unsupported.")
        path.unlink()


def _restore_store(
    backup_root: Path,
    live_root: Path,
    *,
    names: Sequence[str] | None = None,
) -> None:
    selected_backup = backup_root.expanduser().absolute()
    selected_live = live_root.expanduser().absolute()
    if (
        not selected_backup.is_dir()
        or selected_backup.is_symlink()
        or not selected_live.is_dir()
        or selected_live.is_symlink()
    ):
        _fail("A biometric rollback root is unavailable or unsafe.")
    targets = (
        [selected_live / name for name in names]
        if names is not None
        else list(selected_live.iterdir())
    )
    for target in targets:
        if target.exists() or target.is_symlink():
            _remove_entry(target)
    for child in selected_backup.iterdir():
        target = selected_live / child.name
        if child.is_symlink():
            _fail("A biometric rollback backup contains a symlink.")
        if child.is_dir():
            shutil.copytree(child, target, copy_function=shutil.copy2)
            biometric_rehearsal._secure_tree(target)
        elif child.is_file():
            shutil.copy2(child, target)
            biometric_rehearsal._validate_tree_entry(target)
            target.chmod(0o600)
        else:
            _fail("A biometric rollback backup entry is unsupported.")
    selected_live.chmod(0o700)


def _restore_all(
    paths: Mapping[str, Path],
    *,
    live_store_root: Path,
    live_reference_root: Path,
    live_profile_root: Path,
) -> dict[str, Any]:
    _restore_database(
        paths["knowledge_backup_db"], transcript_store.db_path(live_store_root)
    )
    _restore_store(paths["reference_backup"], live_reference_root)
    _restore_store(
        paths["profile_backup"],
        live_profile_root,
        names=biometric_rehearsal.PROFILE_STATE_NAMES,
    )
    return _live_snapshots(
        live_store_root=live_store_root,
        live_reference_root=live_reference_root,
        live_profile_root=live_profile_root,
    )


def _apply_knowledge(
    transition: Mapping[str, Any], *, live_store_root: Path
) -> dict[str, Any]:
    return canonical_rehearsal.apply_reviewed_knowledge_transition(
        transition, store_root=live_store_root
    )


def _apply_biometrics(
    transition: Mapping[str, Any],
    *,
    live_reference_root: Path,
    live_profile_root: Path,
    adapters: Mapping[str, verification.VerificationAdapter] | None,
    test_mode: bool,
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    return biometric_rehearsal.apply_reviewed_biometric_transition(
        transition,
        reference_root=live_reference_root,
        profile_root=live_profile_root,
        reference_baseline=baseline["references"],
        profile_baseline=baseline["profiles"],
        adapters=adapters,
        test_mode=test_mode,
    )


def _validate_test_targets(
    *,
    live_store_root: Path,
    live_reference_root: Path,
    live_profile_root: Path,
) -> None:
    forbidden = {
        Path("~/.transcripts").expanduser().absolute(),
        references.DEFAULT_RUNTIME_ROOT.expanduser().absolute(),
        verification.DEFAULT_RUNTIME_ROOT.expanduser().absolute(),
    }
    selected = {
        live_store_root.expanduser().absolute(),
        live_reference_root.expanduser().absolute(),
        live_profile_root.expanduser().absolute(),
    }
    if any(
        candidate == protected or candidate.is_relative_to(protected)
        for candidate in selected
        for protected in forbidden
    ):
        _fail("Test-mode live apply cannot target a production state root.")


def _validate_terminal_receipt(
    receipt: Mapping[str, Any],
    *,
    paths: Mapping[str, Path],
    current_snapshots: Mapping[str, Any],
    service_state: Mapping[str, Any],
) -> str:
    content_sha256 = str(receipt.get("content_sha256") or "")
    core = {key: value for key, value in receipt.items() if key != "content_sha256"}
    authority_path = Path(str(receipt.get("authority_path") or ""))
    require_private_file(authority_path, paths["root"])
    if (
        receipt.get("schema_version") != LIVE_APPLY_RECEIPT_SCHEMA
        or canonical_artifact_hash(core) != content_sha256
        or receipt.get("current_snapshot_sha256s")
        != _snapshot_hashes(current_snapshots)
        or receipt.get("service_state_after") != service_state
        or receipt.get("authority_file_sha256") != sha256_file(authority_path)
        or receipt.get("prepared_file_sha256") != sha256_file(paths["prepared"])
    ):
        _fail("The Plan 0063 live apply receipt replay drifted.")
    return content_sha256


def replay_live_apply(
    transition_sha256: str,
    *,
    live_store_root: Path,
    live_reference_root: Path,
    live_profile_root: Path,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    service_controller: ServiceController | None = None,
) -> dict[str, Any]:
    """Replay a terminal success or exact-restored failure without reapplying."""

    paths = _paths(runtime_root, transition_sha256)
    require_private_file(paths["prepared"], paths["root"])
    require_private_file(paths["receipt"], paths["root"])
    receipt = read_private_object(paths["receipt"])
    controller = service_controller or SystemdServiceController()
    current = _live_snapshots(
        live_store_root=live_store_root,
        live_reference_root=live_reference_root,
        live_profile_root=live_profile_root,
    )
    service_state = controller.snapshot()
    receipt_sha256 = _validate_terminal_receipt(
        receipt,
        paths=paths,
        current_snapshots=current,
        service_state=service_state,
    )
    return {
        **receipt,
        "receipt_sha256": receipt_sha256,
        "receipt_path": str(paths["receipt"]),
        "idempotent_replay": True,
    }


def apply_live_transition(
    authority_sha256: str,
    *,
    transition_sha256: str,
    expected_request_sha256: str,
    live_store_root: Path,
    live_reference_root: Path,
    live_profile_root: Path,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    adapters: Mapping[str, verification.VerificationAdapter] | None = None,
    service_controller: ServiceController | None = None,
    test_mode: bool = False,
) -> dict[str, Any]:
    """Apply one exact authorized transition or exact-restore every live store."""

    if not SHA256_RE.fullmatch(str(authority_sha256)):
        _fail("The A1 authority hash is invalid.")
    if (adapters is not None or service_controller is not None) and not test_mode:
        _fail("Custom adapters and service controllers are test-only.")
    if test_mode:
        _validate_test_targets(
            live_store_root=live_store_root,
            live_reference_root=live_reference_root,
            live_profile_root=live_profile_root,
        )
    paths = _paths(runtime_root, transition_sha256)
    if paths["receipt"].exists():
        return replay_live_apply(
            transition_sha256,
            live_store_root=live_store_root,
            live_reference_root=live_reference_root,
            live_profile_root=live_profile_root,
            runtime_root=runtime_root,
            service_controller=service_controller,
        )
    if paths["run"].exists():
        _fail("A partial Plan 0063 live apply already exists.")

    authorized = a1.replay_a1_authorization(
        transition_sha256,
        expected_request_sha256=expected_request_sha256,
        live_store_root=live_store_root,
        live_reference_root=live_reference_root,
        live_profile_root=live_profile_root,
        runtime_root=runtime_root,
    )
    if (
        authorized.get("authority_sha256") != authority_sha256
        or authorized.get("a1_authorized") is not True
        or authorized.get("authorization_scope")
        != "one_exact_plan0063_local_live_apply"
        or authorized.get("authorized_actions") != a1.AUTHORIZED_ACTIONS
        or authorized.get("live_mutation_count") != 0
    ):
        _fail("The supplied A1 authority does not authorize this exact live apply.")
    authority_path = Path(str(authorized.get("authority_path") or ""))
    require_private_file(authority_path, runtime_root.expanduser().absolute())
    transition = _load_transition(runtime_root, transition_sha256)
    controller = service_controller or SystemdServiceController()
    service_before = controller.snapshot()
    if any(
        row.get("active_state") != "active" or row.get("sub_state") != "running"
        for row in service_before.get("services", {}).values()
    ) or set(service_before.get("services", {})) != set(SERVICES):
        _fail("Both exact transcript services must be active before live apply.")

    quiesced = False
    backups_ready = False
    baseline: dict[str, Any] | None = None
    prepared: dict[str, Any] | None = None
    try:
        controller.quiesce()
        quiesced = True
        baseline = _live_snapshots(
            live_store_root=live_store_root,
            live_reference_root=live_reference_root,
            live_profile_root=live_profile_root,
        )
        expected_live_hashes = read_private_object(
            Path(str(authorized["request_path"]))
        )["expected_live_state"]["snapshot_sha256s"]
        if _snapshot_hashes(baseline) != expected_live_hashes:
            _fail("Live state changed between A1 replay and service quiescence.")

        ensure_private_tree(paths["root"], paths["run"])
        ensure_private_tree(paths["root"], paths["backup"])
        backups = _prepare_backups(
            paths,
            live_store_root=live_store_root,
            live_reference_root=live_reference_root,
            live_profile_root=live_profile_root,
        )
        if backups["snapshot_sha256s"] != _snapshot_hashes(baseline):
            _fail("The live apply backups do not match the quiesced baseline.")
        backups_ready = True
        prepared_core = {
            "schema_version": LIVE_APPLY_PREPARED_SCHEMA,
            "status": "prepared_for_one_authorized_apply",
            "transition_sha256": transition_sha256,
            "request_sha256": expected_request_sha256,
            "authority_sha256": authority_sha256,
            "authority_file_sha256": sha256_file(authority_path),
            "baseline_snapshot_sha256s": _snapshot_hashes(baseline),
            "backup_snapshot_sha256s": backups["snapshot_sha256s"],
            "expected_apply_counts": dict(authorized["expected_apply_counts"]),
            "service_state_before": service_before,
            "test_mode": test_mode,
            "live_mutation_count": 0,
        }
        prepared = {
            **prepared_core,
            "content_sha256": canonical_artifact_hash(prepared_core),
        }
        write_immutable_private_json(paths["prepared"], prepared)

        knowledge = _apply_knowledge(
            transition, live_store_root=live_store_root
        )
        biometrics = _apply_biometrics(
            transition,
            live_reference_root=live_reference_root,
            live_profile_root=live_profile_root,
            adapters=adapters,
            test_mode=test_mode,
            baseline=baseline,
        )
        expected_counts = dict(authorized["expected_apply_counts"])
        actual_counts = {
            "canonical_people": knowledge["expected_counts"]["knowledge_people"],
            "slot_bindings": knowledge["expected_counts"][
                "knowledge_source_records"
            ],
            "voice_bindings": int(
                transition.get("metrics", {}).get("active_voice_binding_count") or 0
            ),
            "references": biometrics["reference_count"],
            "profiles": biometrics["profile_count"],
            "sources": biometrics["source_count"],
        }
        if actual_counts != expected_counts:
            _fail("The authorized live apply counts did not reconcile.")
        applied = _live_snapshots(
            live_store_root=live_store_root,
            live_reference_root=live_reference_root,
            live_profile_root=live_profile_root,
        )
        service_after = controller.restore()
        quiesced = False
        core = {
            "schema_version": LIVE_APPLY_RECEIPT_SCHEMA,
            "status": "live_apply_completed",
            "transition_sha256": transition_sha256,
            "request_sha256": expected_request_sha256,
            "authority_sha256": authority_sha256,
            "authority_path": str(authority_path),
            "authority_file_sha256": sha256_file(authority_path),
            "prepared_content_sha256": prepared["content_sha256"],
            "prepared_file_sha256": sha256_file(paths["prepared"]),
            "baseline_snapshot_sha256s": _snapshot_hashes(baseline),
            "backup_snapshot_sha256s": backups["snapshot_sha256s"],
            "current_snapshot_sha256s": _snapshot_hashes(applied),
            "actual_apply_counts": actual_counts,
            "created_references": biometrics["created_references"],
            "created_profiles": biometrics["created_profiles"],
            "knowledge_receipts": {
                "people": knowledge["person_receipts"],
                "observations": knowledge["observation_receipt"],
                "profiles": knowledge["profile_receipt"],
            },
            "service_state_before": service_before,
            "service_state_after": service_after,
            "logical_live_apply_count": 1,
            "logical_live_rollback_count": 0,
            "test_mode": test_mode,
            "a1_authorized": True,
            "live_mutation_count": 0 if test_mode else 1,
            "unauthorized_effect_count": 0,
        }
        receipt = {**core, "content_sha256": canonical_artifact_hash(core)}
        write_immutable_private_json(paths["receipt"], receipt)
        return {
            **receipt,
            "receipt_sha256": receipt["content_sha256"],
            "receipt_path": str(paths["receipt"]),
            "idempotent_replay": False,
        }
    except Exception as exc:
        restored: dict[str, Any] | None = None
        rollback_error: Exception | None = None
        if backups_ready and baseline is not None:
            try:
                controller.quiesce()
                quiesced = True
                restored = _restore_all(
                    paths,
                    live_store_root=live_store_root,
                    live_reference_root=live_reference_root,
                    live_profile_root=live_profile_root,
                )
                if restored != baseline:
                    _fail("The failed live apply did not restore exact baseline state.")
            except Exception as restore_exc:  # pragma: no cover - catastrophic path
                rollback_error = restore_exc
        service_after: dict[str, Any] | None = None
        if quiesced:
            try:
                service_after = controller.restore()
                quiesced = False
            except Exception as service_exc:  # pragma: no cover - catastrophic path
                rollback_error = rollback_error or service_exc
        if paths["run"].exists() and prepared is not None and restored is not None:
            failure_core = {
                "schema_version": LIVE_APPLY_RECEIPT_SCHEMA,
                "status": "live_apply_failed_and_exactly_restored",
                "transition_sha256": transition_sha256,
                "request_sha256": expected_request_sha256,
                "authority_sha256": authority_sha256,
                "authority_path": str(authority_path),
                "authority_file_sha256": sha256_file(authority_path),
                "prepared_content_sha256": prepared["content_sha256"],
                "prepared_file_sha256": sha256_file(paths["prepared"]),
                "baseline_snapshot_sha256s": _snapshot_hashes(baseline),
                "backup_snapshot_sha256s": _snapshot_hashes(baseline),
                "current_snapshot_sha256s": _snapshot_hashes(restored),
                "failure_type": type(exc).__name__,
                "service_state_before": service_before,
                "service_state_after": service_after,
                "logical_live_apply_count": 1,
                "logical_live_rollback_count": 1,
                "test_mode": test_mode,
                "a1_authorized": True,
                "live_mutation_count": 0,
                "unauthorized_effect_count": 0,
            }
            failure = {
                **failure_core,
                "content_sha256": canonical_artifact_hash(failure_core),
            }
            write_immutable_private_json(paths["receipt"], failure)
        if rollback_error is not None:
            raise Plan0063LiveApplyError(
                "Plan 0063 live apply failed and rollback could not be proved."
            ) from rollback_error
        raise Plan0063LiveApplyError(
            "Plan 0063 live apply failed; exact rollback completed when mutation began."
        ) from exc
    finally:
        if quiesced:
            controller.restore()


__all__ = [
    "DEFAULT_RUNTIME_ROOT",
    "LIVE_APPLY_PREPARED_SCHEMA",
    "LIVE_APPLY_RECEIPT_SCHEMA",
    "Plan0063LiveApplyError",
    "SERVICES",
    "ServiceController",
    "SystemdServiceController",
    "apply_live_transition",
    "replay_live_apply",
]
