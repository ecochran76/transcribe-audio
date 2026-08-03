"""Plan 0053 G2 sealed positive holdout and adversarial validation."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import acoustic_content_preservation as preservation
import acoustic_content_preservation_adversarial as adversarial
import acoustic_generation5_diagnostic_authority as g0
import acoustic_generation5_j1_acceptance as j1
from acoustic_audio_derivatives import (
    ensure_private_tree,
    require_private_file,
    sha256_file,
    write_immutable_private_json,
)


PREVIEW_SCHEMA = "transcribe-audio.generation5-holdout-preview.v1"
MANIFEST_SCHEMA = "transcribe-audio.generation5-holdout-manifest.v1"
RECEIPT_SCHEMA = "transcribe-audio.generation5-holdout-receipt.v1"
REPLAY_SCHEMA = "transcribe-audio.generation5-holdout-replay.v1"
DEFAULT_RUNTIME_ROOT = Path("~/.local/state/transcribe-audio/plan-0053/g2")
G0_PREVIEW_SHA256 = j1.g1.G0_PREVIEW_SHA256
G0_MANIFEST_SHA256 = j1.g1.G0_MANIFEST_SHA256
J1_PREVIEW_SHA256 = "1ccb0b9b747760b6202827f849d2575956a67ad03dd3e374d169b1777292eeda"
J1_MANIFEST_SHA256 = "f7fea046193fe942f6f9ea78189cd8006b6b7ca50c95464f5fa8bfc6604088d8"
CONTRACT_SHA256 = j1.G1_CONTRACT_SHA256
BOUND_MODULES = (
    "acoustic_content_preservation.py",
    "acoustic_content_preservation_adversarial.py",
    "acoustic_generation5_holdout.py",
)


class Generation5HoldoutError(ValueError):
    """Raised when G2 holdout validation cannot remain sealed and exact."""


def _canonical_hash(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Generation5HoldoutError("Private authority is unreadable.") from exc
    if not isinstance(value, dict):
        raise Generation5HoldoutError("Private authority must be an object.")
    return value


def _git(arguments: Sequence[str], *, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", *arguments], cwd=Path(__file__).resolve().parent,
        capture_output=True, text=not binary, check=False,
    )
    if result.returncode:
        raise Generation5HoldoutError("Repository authority is unavailable.")
    return result.stdout if binary else result.stdout.strip()


def _repository_authority() -> dict[str, Any]:
    if _git(["status", "--porcelain=v1", "--untracked-files=normal"]):
        raise Generation5HoldoutError("Repository must be clean.")
    if str(_git(["rev-list", "--left-right", "--count", "@{upstream}...HEAD"])).split() != ["0", "0"]:
        raise Generation5HoldoutError("Repository must be upstream-even.")
    commit = str(_git(["rev-parse", "HEAD"]))
    if not re.fullmatch(r"[a-f0-9]{40}", commit):
        raise Generation5HoldoutError("Repository commit is invalid.")
    module_hashes = {}
    for name in BOUND_MODULES:
        body = _git(["show", f"{commit}:{name}"], binary=True)
        if not isinstance(body, bytes):
            raise Generation5HoldoutError("Committed module body is unavailable.")
        digest = hashlib.sha256(body).hexdigest()
        if digest != sha256_file(Path(__file__).resolve().parent / name):
            raise Generation5HoldoutError("Committed module body drifted.")
        module_hashes[name] = digest
    return {
        "commit": commit,
        "module_sha256": module_hashes,
        "clean": True,
        "upstream_ahead": 0,
        "upstream_behind": 0,
    }


def _parent_authorities() -> tuple[dict[str, Any], dict[str, Any]]:
    g0_paths = g0._paths(g0.DEFAULT_RUNTIME_ROOT, G0_PREVIEW_SHA256)
    j1_paths = j1._paths(j1.DEFAULT_RUNTIME_ROOT, J1_PREVIEW_SHA256)
    for paths in (g0_paths, j1_paths):
        require_private_file(paths["manifest"], paths["root"].expanduser().absolute())
    if sha256_file(g0_paths["manifest"]) != G0_MANIFEST_SHA256 or sha256_file(j1_paths["manifest"]) != J1_MANIFEST_SHA256:
        raise Generation5HoldoutError("Parent manifest drifted.")
    g0_preview = _read_json(g0_paths["manifest"])["preview"]
    j1_preview = _read_json(j1_paths["manifest"])["preview"]
    if (
        g0_preview.get("content_sha256") != G0_PREVIEW_SHA256
        or g0_preview.get("did_measure_holdout") is not False
        or j1_preview.get("content_sha256") != J1_PREVIEW_SHA256
        or j1_preview.get("contract_sha256") != CONTRACT_SHA256
        or j1_preview.get("status") != "accepted_for_g2_only"
        or j1_preview.get("review_decision") != "PASS"
        or j1_preview.get("action_vector", {}).get("run_g2_positive_holdout_once") is not True
    ):
        raise Generation5HoldoutError("Parent authority drifted.")
    return dict(g0_preview), dict(j1_preview)


def _collect_holdout(g0_preview: Mapping[str, Any]) -> list[dict[str, Any]]:
    private = g0_preview.get("private_evidence")
    members = private.get("holdout") if isinstance(private, Mapping) else None
    if not isinstance(members, list) or len(members) != 7:
        raise Generation5HoldoutError("Positive holdout membership is invalid.")
    results = []
    for member in members:
        measurement = preservation.measure(
            Path(str(member.get("path") or "")),
            expected_source_sha256=str(member.get("source_sha256") or ""),
            channel_policy_authority_sha256=J1_PREVIEW_SHA256,
        )
        results.append(
            {
                "source_sha256": member.get("source_sha256"),
                "authority_origin": member.get("authority_origin"),
                "measurement": measurement,
            }
        )
    return results


def _collect_heldout_adversaries(g0_preview: Mapping[str, Any]) -> dict[str, Any]:
    members = sorted(
        g0_preview["private_evidence"]["holdout"],
        key=lambda member: str(member.get("source_sha256") or ""),
    )
    source = members[0]
    return adversarial.run_holdout_adversaries(
        Path(str(source["path"])),
        expected_source_sha256=str(source["source_sha256"]),
        channel_policy_authority_sha256=J1_PREVIEW_SHA256,
    )


def _validate_holdout(results: list[dict[str, Any]]) -> None:
    if len(results) != 7 or len({item.get("source_sha256") for item in results}) != 7:
        raise Generation5HoldoutError("Positive holdout denominator is invalid.")
    for item in results:
        measurement = item.get("measurement")
        if not isinstance(measurement, Mapping):
            raise Generation5HoldoutError("Positive holdout measurement is invalid.")
        if (
            measurement.get("status") != "passing"
            or measurement.get("reason_codes") != []
            or measurement.get("output_sample_error") not in {-1, 0, 1}
            or (measurement.get("recipe_reference_decode") or {}).get("pcm_sha256")
            != (measurement.get("production_wav") or {}).get("pcm_sha256")
        ):
            raise Generation5HoldoutError("Positive holdout did not pass the frozen rule.")


def preview_generation5_holdout(
    *,
    g0_preview: Mapping[str, Any] | None = None,
    j1_preview: Mapping[str, Any] | None = None,
    holdout_results: list[dict[str, Any]] | None = None,
    heldout_adversarial: Mapping[str, Any] | None = None,
    repository_authority: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if g0_preview is None or j1_preview is None:
        parent_g0, parent_j1 = _parent_authorities()
    else:
        parent_g0, parent_j1 = dict(g0_preview), dict(j1_preview)
    if (
        parent_g0.get("content_sha256") != G0_PREVIEW_SHA256
        or parent_j1.get("content_sha256") != J1_PREVIEW_SHA256
        or parent_j1.get("status") != "accepted_for_g2_only"
    ):
        raise Generation5HoldoutError("G2 parent authority is invalid.")
    results = list(holdout_results if holdout_results is not None else _collect_holdout(parent_g0))
    negative = dict(heldout_adversarial or _collect_heldout_adversaries(parent_g0))
    _validate_holdout(results)
    if (
        negative.get("seed") != adversarial.HOLDOUT_SEED
        or negative.get("case_count") != 11
        or negative.get("all_expected_rejections_observed") is not True
    ):
        raise Generation5HoldoutError("Held-out adversarial family did not pass.")
    public_negative = {
        key: value for key, value in negative.items()
        if key not in {"private_fixture_hashes", "private_case_measurements"}
    }
    actions = {
        "submit_to_j2": True,
        "enumerate_generation5_candidates": False,
        "reveal_gold": False,
        "run_predictions_or_models": False,
        "mutate_profiles_or_references": False,
        "enable_default_integration": False,
        "run_historical_reprocessing": False,
    }
    private = {
        "positive_holdout_measurements": results,
        "heldout_adversarial": negative,
    }
    core = {
        "schema_version": PREVIEW_SCHEMA,
        "status": "ready_for_independent_j2_audit",
        "repository_authority": dict(repository_authority or _repository_authority()),
        "g0_preview_sha256": G0_PREVIEW_SHA256,
        "j1_preview_sha256": J1_PREVIEW_SHA256,
        "contract_sha256": CONTRACT_SHA256,
        "positive_holdout_count": len(results),
        "positive_holdout_pass_count": sum(item["measurement"]["status"] == "passing" for item in results),
        "positive_holdout_results_sha256": _canonical_hash(results),
        "heldout_adversarial": public_negative,
        "action_vector": actions,
        "private_evidence": private,
        "contains_paths": False,
        "contains_private_membership": True,
        "did_measure_holdout": True,
        "did_instantiate_heldout_negative_family": True,
        "did_access_gold": False,
        "did_load_or_run_models": False,
    }
    return {**core, "content_sha256": _canonical_hash(core)}


def _portable(preview: Mapping[str, Any]) -> dict[str, Any]:
    value = {
        key: item for key, item in preview.items()
        if key not in {"private_evidence", "repository_authority"}
    }
    value["schema_version"] = RECEIPT_SCHEMA
    value["contains_private_membership"] = False
    return value


def _paths(runtime_root: Path, content_sha256: str) -> dict[str, Path]:
    root = runtime_root.expanduser().absolute()
    run = root / f"generation5-holdout-{content_sha256[:24]}"
    return {
        "root": root,
        "run": run,
        "manifest": run / "private-manifest.json",
        "receipt": run / "receipt.json",
    }


def apply_generation5_holdout(
    reviewed_preview: Mapping[str, Any], *, expected_content_sha256: str,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
) -> dict[str, Any]:
    preview = preview_generation5_holdout()
    if dict(reviewed_preview) != preview or preview["content_sha256"] != expected_content_sha256:
        raise Generation5HoldoutError("Reviewed G2 preview is stale.")
    paths = _paths(runtime_root, expected_content_sha256)
    if paths["manifest"].exists() or paths["receipt"].exists():
        return replay_generation5_holdout(expected_content_sha256, runtime_root=runtime_root)
    ensure_private_tree(paths["root"], paths["run"])
    manifest = {"schema_version": MANIFEST_SCHEMA, "status": "frozen", "preview": preview}
    write_immutable_private_json(paths["manifest"], manifest)
    receipt = {
        **_portable(preview), "manifest_sha256": sha256_file(paths["manifest"]), "mode": "0600"
    }
    write_immutable_private_json(paths["receipt"], receipt)
    return {**receipt, "idempotent_replay": False}


def replay_generation5_holdout(
    expected_content_sha256: str, *, runtime_root: Path = DEFAULT_RUNTIME_ROOT,
) -> dict[str, Any]:
    preview = preview_generation5_holdout()
    if preview["content_sha256"] != expected_content_sha256:
        raise Generation5HoldoutError("G2 authority drifted.")
    paths = _paths(runtime_root, expected_content_sha256)
    require_private_file(paths["manifest"], paths["root"])
    require_private_file(paths["receipt"], paths["root"])
    manifest = _read_json(paths["manifest"])
    receipt = _read_json(paths["receipt"])
    expected_manifest = {"schema_version": MANIFEST_SCHEMA, "status": "frozen", "preview": preview}
    expected_receipt = {
        **_portable(preview), "manifest_sha256": sha256_file(paths["manifest"]), "mode": "0600"
    }
    if manifest != expected_manifest or receipt != expected_receipt:
        raise Generation5HoldoutError("G2 body or receipt drifted.")
    return {**receipt, "replay_schema_version": REPLAY_SCHEMA, "idempotent_replay": True}
