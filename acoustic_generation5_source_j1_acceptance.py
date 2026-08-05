"""Freeze Plan 0055 independent J1 cohort and private-gold acceptance."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import acoustic_generation5_source_gold as gold
from acoustic_audio_derivatives import (
    ensure_private_tree,
    require_private_file,
    sha256_file,
    write_immutable_private_json,
)


PREVIEW_SCHEMA = "transcribe-audio.generation5-source-j1-preview.v1"
MANIFEST_SCHEMA = "transcribe-audio.generation5-source-j1-manifest.v1"
RECEIPT_SCHEMA = "transcribe-audio.generation5-source-j1-receipt.v1"
REPLAY_SCHEMA = "transcribe-audio.generation5-source-j1-replay.v1"
GOLD_PREVIEW_SHA256 = "9cd1a5c41920de2f0dc562c868268c6eaa9091be9cd7e88794969e460858f971"
GOLD_MANIFEST_SHA256 = "b4cadac5f76d3279f9c48ae7559fc37ab3071be043dcc889c8a92c7b6b21cde5"
REVIEWER_HANDLE = "/root/plan0055_j0_review"
DEFAULT_RUNTIME_ROOT = Path("~/.local/state/transcribe-audio/plan-0055/j1")
MODULE_NAME = Path(__file__).name


class Generation5SourceJ1Error(ValueError):
    """Raised when the independent J1 acceptance cannot be bound exactly."""


def _canonical_hash(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(body.encode()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Generation5SourceJ1Error("Private J1 authority is unreadable.") from exc
    if not isinstance(value, dict):
        raise Generation5SourceJ1Error("Private J1 authority must be an object.")
    return value


def _git(arguments: Sequence[str], *, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", *arguments], cwd=Path(__file__).resolve().parent,
        capture_output=True, text=not binary, check=False,
    )
    if result.returncode:
        raise Generation5SourceJ1Error("Repository authority is unavailable.")
    return result.stdout if binary else result.stdout.strip()


def _repository_authority() -> dict[str, Any]:
    if _git(["status", "--porcelain=v1", "--untracked-files=normal"]):
        raise Generation5SourceJ1Error("Repository must be clean.")
    if str(_git(["rev-list", "--left-right", "--count", "@{upstream}...HEAD"])).split() != ["0", "0"]:
        raise Generation5SourceJ1Error("Repository must be upstream-even.")
    commit = str(_git(["rev-parse", "HEAD"]))
    body = _git(["show", f"{commit}:{MODULE_NAME}"], binary=True)
    if not re.fullmatch(r"[a-f0-9]{40}", commit) or not isinstance(body, bytes) or hashlib.sha256(body).hexdigest() != sha256_file(Path(__file__).resolve()):
        raise Generation5SourceJ1Error("Committed J1 module drifted.")
    return {"commit": commit, "module_sha256": hashlib.sha256(body).hexdigest(),
            "clean": True, "upstream_ahead": 0, "upstream_behind": 0}


def _gold_proposal() -> dict[str, Any]:
    replay = gold.replay_generation5_source_gold(GOLD_PREVIEW_SHA256)
    paths = gold._paths(gold.DEFAULT_RUNTIME_ROOT, GOLD_PREVIEW_SHA256)
    if replay.get("idempotent_replay") is not True or sha256_file(paths["manifest"]) != GOLD_MANIFEST_SHA256:
        raise Generation5SourceJ1Error("Gold proposal authority drifted.")
    proposal = _read_json(paths["manifest"]).get("preview")
    if not isinstance(proposal, dict):
        raise Generation5SourceJ1Error("Gold proposal is missing.")
    private = proposal.get("private_gold")
    selected = private.get("selected_cases") if isinstance(private, Mapping) else None
    all_cases = private.get("all_cases") if isinstance(private, Mapping) else None
    selected_ordinals = [int(case.get("enumerated_ordinal") or 0) for case in selected or [] if isinstance(case, Mapping)]
    population = proposal.get("population_result")
    actions = proposal.get("action_vector")
    if (
        proposal.get("content_sha256") != GOLD_PREVIEW_SHA256
        or proposal.get("status") != "ready_for_independent_j1_review"
        or proposal.get("reviewed_candidate_count") != 12
        or proposal.get("reviewed_speaker_label_count") != 40
        or proposal.get("operator_supplied_answer_count") != 39
        or proposal.get("context_derived_answer_count") != 1
        or proposal.get("required_ordinals") != [1, 2]
        or proposal.get("combinations_checked") != 1
        or proposal.get("population_feasible") is not True
        or not isinstance(population, Mapping)
        or population.get("passing") is not True
        or selected_ordinals != [1, 2, 3, 4, 5, 6, 7]
        or not isinstance(all_cases, list) or len(all_cases) != 12
        or not isinstance(actions, Mapping)
        or actions.get("submit_exact_population_and_gold_to_j1") is not True
        or any(actions.get(key) is not False for key in (
            "freeze_cohort_or_gold", "run_models_or_predictions", "reveal_gold_to_workers",
            "mutate_profiles_or_references", "enable_default_integration", "run_historical_reprocessing",
        ))
    ):
        raise Generation5SourceJ1Error("Gold proposal did not satisfy J1 gates.")
    recomputed, recomputed_population, checked = gold.select_first_passing(
        all_cases, {str(case.get("source_sha256") or "") for case in all_cases}
    )
    if recomputed != selected or recomputed_population != population or checked != 1:
        raise Generation5SourceJ1Error("Gold proposal selection is not reproducible.")
    return proposal


def preview_generation5_source_j1_acceptance(
    *, gold_proposal: Mapping[str, Any] | None = None,
    repository_authority: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    proposal = dict(gold_proposal or _gold_proposal())
    private = proposal.get("private_gold")
    selected = private.get("selected_cases") if isinstance(private, Mapping) else None
    if not isinstance(selected, list) or len(selected) != 7:
        raise Generation5SourceJ1Error("Selected private gold is unavailable.")
    findings = {
        "complete_twelve_case_forty_label_denominator": True,
        "alias_canonicalization_recomputed": True,
        "mark_mba_wright_context_authority_supported": True,
        "mandatory_required_ordinals_retained": True,
        "first_lexicographic_combination_selected": True,
        "seven_distinct_sources_transcripts_recordings_conversations": True,
        "minimum_five_people": True,
        "minimum_four_same_person_session_pairs": True,
        "both_enrolled_people_have_two_recordings": True,
        "zero_prior_overlap": True,
        "private_custody_and_replay_passed": True,
        "identity_or_acoustic_models_used_during_gold_review": False,
    }
    actions = {
        "freeze_selected_cohort_and_private_gold": True,
        "prepare_gold_blind_paired_workers": True,
        "run_models_or_predictions": False,
        "reveal_gold_to_workers": False,
        "mutate_profiles_or_references": False,
        "enable_default_integration": False,
        "run_historical_reprocessing": False,
    }
    core = {
        "schema_version": PREVIEW_SCHEMA, "status": "accepted_for_gold_freeze_and_blind_worker_preparation",
        "review_decision": "PASS", "reviewer_handle": REVIEWER_HANDLE,
        "repository_authority": dict(repository_authority or _repository_authority()),
        "gold_preview_sha256": GOLD_PREVIEW_SHA256, "gold_manifest_sha256": GOLD_MANIFEST_SHA256,
        "findings": findings, "findings_sha256": _canonical_hash(findings),
        "selected_case_ids_sha256": proposal["selected_case_ids_sha256"],
        "selected_source_set_sha256": proposal["selected_source_set_sha256"],
        "selected_transcript_set_sha256": proposal["selected_transcript_set_sha256"],
        "population_result": proposal["population_result"],
        "private_gold": {"selected_cases": selected, "selected_case_ids": private["selected_case_ids"]},
        "action_vector": actions, "contains_private_gold": True,
        "did_freeze_cohort_or_gold": False, "did_run_models_or_predictions": False,
        "did_reveal_gold_to_workers": False,
    }
    return {**core, "content_sha256": _canonical_hash(core)}


def _portable(preview: Mapping[str, Any]) -> dict[str, Any]:
    return {key: preview[key] for key in (
        "status", "review_decision", "reviewer_handle", "gold_preview_sha256",
        "gold_manifest_sha256", "findings_sha256", "selected_case_ids_sha256",
        "selected_source_set_sha256", "selected_transcript_set_sha256", "population_result",
        "action_vector",
    )}


def _paths(runtime_root: Path, content_sha256: str) -> dict[str, Path]:
    root = runtime_root.expanduser().absolute()
    run = root / f"generation5-source-j1-{content_sha256[:24]}"
    return {"root": root, "run": run, "manifest": run / "private-gold-manifest.json", "receipt": run / "receipt.json"}


def apply_generation5_source_j1_acceptance(
    reviewed_preview: Mapping[str, Any], *, expected_content_sha256: str,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
) -> dict[str, Any]:
    preview = preview_generation5_source_j1_acceptance()
    if dict(reviewed_preview) != preview or preview["content_sha256"] != expected_content_sha256:
        raise Generation5SourceJ1Error("Reviewed J1 acceptance is stale.")
    paths = _paths(runtime_root, expected_content_sha256)
    if paths["receipt"].exists():
        return replay_generation5_source_j1_acceptance(expected_content_sha256, runtime_root=runtime_root)
    ensure_private_tree(paths["root"], paths["run"])
    frozen = dict(preview)
    frozen["did_freeze_cohort_or_gold"] = True
    manifest = {"schema_version": MANIFEST_SCHEMA, "status": "private_gold_frozen_not_revealed", "preview": frozen}
    write_immutable_private_json(paths["manifest"], manifest)
    receipt = {**_portable(preview), "schema_version": RECEIPT_SCHEMA,
               "preview_content_sha256": expected_content_sha256,
               "manifest_sha256": sha256_file(paths["manifest"]), "mode": "0600",
               "did_freeze_cohort_or_gold": True, "did_run_models_or_predictions": False,
               "did_reveal_gold_to_workers": False}
    write_immutable_private_json(paths["receipt"], receipt)
    return {**receipt, "idempotent_replay": False}


def replay_generation5_source_j1_acceptance(
    expected_content_sha256: str, *, runtime_root: Path = DEFAULT_RUNTIME_ROOT,
) -> dict[str, Any]:
    paths = _paths(runtime_root, expected_content_sha256)
    require_private_file(paths["manifest"], paths["root"])
    require_private_file(paths["receipt"], paths["root"])
    manifest, receipt = _read_json(paths["manifest"]), _read_json(paths["receipt"])
    preview = manifest.get("preview")
    if not isinstance(preview, Mapping):
        raise Generation5SourceJ1Error("Frozen private gold is missing.")
    original = dict(preview)
    original["did_freeze_cohort_or_gold"] = False
    core = {key: value for key, value in original.items() if key != "content_sha256"}
    expected_receipt = {**_portable(original), "schema_version": RECEIPT_SCHEMA,
                        "preview_content_sha256": expected_content_sha256,
                        "manifest_sha256": sha256_file(paths["manifest"]), "mode": "0600",
                        "did_freeze_cohort_or_gold": True, "did_run_models_or_predictions": False,
                        "did_reveal_gold_to_workers": False}
    if (_canonical_hash(core) != expected_content_sha256 or original.get("content_sha256") != expected_content_sha256
            or manifest != {"schema_version": MANIFEST_SCHEMA, "status": "private_gold_frozen_not_revealed", "preview": preview}
            or receipt != expected_receipt):
        raise Generation5SourceJ1Error("Frozen J1 authority drifted.")
    return {**receipt, "replay_schema_version": REPLAY_SCHEMA, "idempotent_replay": True}
