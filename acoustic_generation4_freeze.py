"""Plan 0052 G2 immutable cohort, gold commitment, and policy envelope."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import acoustic_generation4_acoustic_contract as acoustic
import acoustic_generation4_cohort as cohort
import acoustic_generation4_context_contract as context
from acoustic_audio_derivatives import (
    ensure_private_tree,
    require_private_file,
    sha256_file,
    write_immutable_private_json,
)


PREVIEW_SCHEMA = "transcribe-audio.generation4-g2-freeze-preview.v1"
MANIFEST_SCHEMA = "transcribe-audio.generation4-g2-freeze-manifest.v1"
RECEIPT_SCHEMA = "transcribe-audio.generation4-g2-freeze-receipt.v1"
REPLAY_SCHEMA = "transcribe-audio.generation4-g2-freeze-replay.v1"
DEFAULT_RUNTIME_ROOT = Path("~/.local/state/transcribe-audio/plan-0052/g2")
G1A_PREVIEW_SHA256 = (
    "9648201f4d3e70f65d396bb1fe82fb9aad57603ff077ccd99b7d61532a0889d7"
)
G1A_MANIFEST_SHA256 = (
    "fed08c49b26024b041774b9df7b067ae9d156022fd8a7a16dbf0df8b85451c0f"
)
GOLD_CONTENT_SHA256 = (
    "37f3a2da83cdbecaa936fbad477490c234c3d25ba55243f68a413f06dee4557a"
)
GOLD_FILE_SHA256 = (
    "5b43119baddb24c794ebfe3224b735d182bc7b10e374166d71f68c9ea61ef65d"
)
G1B_CONTENT_SHA256 = (
    "eae21ec7842803a8cf6aa695b5146927ee9da33e2133ab542cd446fcdc039aab"
)
G1C_CONTENT_SHA256 = (
    "f539146dfccc3a8025d20713b5cf02762d7d5a5d25cb01f4886f6dedda44bb18"
)
CALIBRATION_ROOT = Path(
    "~/.local/state/transcribe-audio/plan-0037/generation-3/"
    "recalibration-executions/generation3-recalibration-execution-"
    "39298c74aab4a773945268cd"
)
CALIBRATION_APPLICATION_SHA256 = (
    "308f326d3fe9baa175ed32c90df4255a8d4bfc1924c6f925eab490ae2832f4d1"
)
CALIBRATION_THRESHOLD_SET_SHA256 = (
    "a927b0d9752d4b79ec42f5248afd2028db1c44414ff2d733c46c7b01b6d16759"
)
CALIBRATION_SCORE_MATRIX_SHA256 = (
    "3fb983b06b1984724c2f0e3e3c01f55065ff755e36416260c33fe0f2649201c2"
)


class Generation4FreezeError(ValueError):
    """Raised when G2 cannot freeze exact pre-model authority."""


def _canonical_hash(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(body.encode()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Generation4FreezeError("Frozen private authority is unreadable.") from exc
    if not isinstance(value, dict):
        raise Generation4FreezeError("Frozen private authority must be an object.")
    return value


def _repository_authority() -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=normal"],
        cwd=root, capture_output=True, text=True, check=False,
    )
    parity = subprocess.run(
        ["git", "rev-list", "--left-right", "--count", "@{upstream}...HEAD"],
        cwd=root, capture_output=True, text=True, check=False,
    )
    if status.returncode or status.stdout or parity.returncode or parity.stdout.split() != ["0", "0"]:
        raise Generation4FreezeError("Repository must be clean and upstream-even.")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()
    return {
        "commit": commit,
        "module_sha256": sha256_file(Path(__file__).resolve()),
        "clean": True,
        "upstream_ahead": 0,
        "upstream_behind": 0,
    }


def _g1a() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    paths = cohort._paths(cohort.DEFAULT_RUNTIME_ROOT, G1A_PREVIEW_SHA256)
    require_private_file(paths["manifest"], paths["root"].expanduser().absolute())
    manifest = _read_json(paths["manifest"])
    preview = manifest.get("preview")
    if (
        sha256_file(paths["manifest"]) != G1A_MANIFEST_SHA256
        or not isinstance(preview, Mapping)
        or preview.get("content_sha256") != G1A_PREVIEW_SHA256
        or preview.get("status") != "passing_population_proposal"
        or preview.get("population", {}).get("passing") is not True
        or preview.get("action_vector", {}).get("submit_population_proposal_to_j1") is not True
        or preview.get("did_freeze_cohort_or_gold") is not False
    ):
        raise Generation4FreezeError("G1A passing authority drifted.")
    proposed = preview.get("private_evidence", {}).get("proposed_original_cohort")
    if not isinstance(proposed, list) or len(proposed) != 7:
        raise Generation4FreezeError("G1A exact cohort is unavailable.")
    return dict(preview), [dict(item) for item in proposed if isinstance(item, Mapping)]


def _gold(expected_cases: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = (
        Path.home()
        / ".local/state/transcribe-audio/plan-0052/g1a/operator-gold/"
        f"generation4-private-gold-{GOLD_CONTENT_SHA256[:24]}/private-gold.json"
    )
    require_private_file(path, path.parents[2])
    gold = _read_json(path)
    if (
        sha256_file(path) != GOLD_FILE_SHA256
        or gold.get("content_sha256") != GOLD_CONTENT_SHA256
        or gold.get("schema_version") != cohort.GOLD_SCHEMA
        or gold.get("status") != "private_operator_review_complete_not_frozen"
        or gold.get("did_reveal_gold_to_prediction_workers") is not False
        or gold.get("did_freeze_cohort_or_gold") is not False
    ):
        raise Generation4FreezeError("Private gold commitment authority drifted.")
    by_key = {
        (str(item.get("source_sha256")), str(item.get("transcript_sha256"))): dict(item)
        for item in gold.get("cases", [])
        if isinstance(item, Mapping)
    }
    selected = []
    for expected in expected_cases:
        key = (str(expected.get("source_sha256")), str(expected.get("transcript_sha256")))
        actual = by_key.get(key)
        if actual is None or actual != dict(expected):
            raise Generation4FreezeError("G1A cohort and private gold disagree.")
        selected.append(actual)
    return gold, selected


def _g1b() -> dict[str, Any]:
    root = CALIBRATION_ROOT.expanduser().absolute()
    application_path = root / "threshold-application.json"
    receipt_path = root / "threshold-receipt.json"
    require_private_file(application_path, root.parent)
    require_private_file(receipt_path, root.parent)
    application = _read_json(application_path)
    receipt = _read_json(receipt_path)
    if (
        sha256_file(application_path) != CALIBRATION_APPLICATION_SHA256
        or receipt.get("threshold_application_sha256") != CALIBRATION_APPLICATION_SHA256
        or receipt.get("threshold_set_sha256") != CALIBRATION_THRESHOLD_SET_SHA256
        or receipt.get("score_matrix_sha256") != CALIBRATION_SCORE_MATRIX_SHA256
        or application.get("threshold_unit_count") != 9
    ):
        raise Generation4FreezeError("Calibration-only authority drifted.")
    packet = {
        "split": "calibration",
        "threshold_unit_count": 9,
        "selection_objective": application["selection_objective"],
        "threshold_application_sha256": CALIBRATION_APPLICATION_SHA256,
        "threshold_set_sha256": CALIBRATION_THRESHOLD_SET_SHA256,
        "score_matrix_sha256": CALIBRATION_SCORE_MATRIX_SHA256,
        "thresholds": application["thresholds"],
        "did_read_generation4_gold": False,
        "did_read_generation4_holdout": False,
        "did_load_or_run_models": False,
    }
    return acoustic.replay_generation4_acoustic_contract(
        packet, expected_content_sha256=G1B_CONTENT_SHA256
    )


def _j1_acceptance() -> dict[str, Any]:
    core = {
        "schema_version": "transcribe-audio.generation4-j1-acceptance.v1",
        "status": "signed_design_acceptance",
        "signed_by": "/root/recalibration_authority_reaudit",
        "signed_date": "2026-08-03",
        "g1a_preview_sha256": G1A_PREVIEW_SHA256,
        "g1a_manifest_sha256": G1A_MANIFEST_SHA256,
        "g1b_content_sha256": G1B_CONTENT_SHA256,
        "g1c_content_sha256": G1C_CONTENT_SHA256,
        "authority_rework_closed": True,
        "authorize_g2_policy_envelope_freeze": True,
        "authorize_g2_exact_cohort_freeze": True,
        "authorize_g2_private_gold_commitment_freeze": True,
        "authorize_gold_reveal_or_models": False,
    }
    return {**core, "content_sha256": _canonical_hash(core)}


def _terminal_policy() -> dict[str, Any]:
    return {
        "precedence": ["stop", "reject_acoustic_factor", "advance_to_limited_pilot_plan", "keep_shadow_and_refine"],
        "stop_on": ["authority_drift", "blindness_breach", "privacy_failure", "incomplete_denominators", "exhausted_attempts", "replay_failure", "safety_invalid_run"],
        "reject_on": ["added_high_confidence_wrong_identity", "reduced_assignment_correctness", "reduced_candidate_recall"],
        "advance_requires": ["all_gates_pass", "correctness_not_worse", "recall_not_worse", "zero_augmented_high_confidence_wrong", "fix_one_baseline_error_or_two_safe_review_conversions"],
        "default_valid_outcome": "keep_shadow_and_refine",
        "terminal_outcomes_authorize_production_mutation": False,
    }


def preview_generation4_freeze(
    *, repository_authority: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    g1a, proposed = _g1a()
    gold, selected_gold = _gold(proposed)
    g1b = _g1b()
    g1c = context.build_generation4_context_contract()
    if g1c["content_sha256"] != G1C_CONTENT_SHA256:
        raise Generation4FreezeError("G1C contextual contract drifted.")
    population = cohort.evaluate_population(
        selected_gold,
        expected_sources={str(item["source_sha256"]) for item in proposed},
    )
    if not population["passing"]:
        raise Generation4FreezeError("Frozen cohort population no longer passes.")
    actions = {
        "run_g3_blind_preparation_and_context_baseline": True,
        "reveal_gold": False,
        "send_augmented_prediction_turn": False,
        "load_or_run_acoustic_models": False,
        "score": False,
        "mutate_profiles_or_references": False,
        "enable_default_integration": False,
        "run_historical_reprocessing": False,
    }
    private = {
        "cohort_membership": proposed,
        "frozen_gold_cases": selected_gold,
        "calibration_thresholds": g1b,
    }
    core = {
        "schema_version": PREVIEW_SCHEMA,
        "status": "immutable_pre_model_authority",
        "repository_authority": dict(repository_authority or _repository_authority()),
        "j1_acceptance": _j1_acceptance(),
        "g1a_preview_sha256": G1A_PREVIEW_SHA256,
        "g1a_manifest_sha256": G1A_MANIFEST_SHA256,
        "source_authority": dict(g1a["authority"]),
        "cohort_count": len(proposed),
        "cohort_set_sha256": _canonical_hash(sorted(str(item["source_sha256"]) for item in proposed)),
        "population": population,
        "gold_content_sha256": gold["content_sha256"],
        "gold_file_sha256": GOLD_FILE_SHA256,
        "gold_commitment_sha256": _canonical_hash(selected_gold),
        "gold_case_count": len(selected_gold),
        "g1b_content_sha256": g1b["content_sha256"],
        "selected_factor_contract_sha256": g1b["selected_factor_contract_sha256"],
        "full_matrix_unit_count": g1b["full_matrix_unit_count"],
        "full_matrix_unit_set_sha256": g1b["full_matrix_unit_set_sha256"],
        "acoustic_contract_hashes": dict(g1b["contract_hashes"]),
        "g1c_content_sha256": g1c["content_sha256"],
        "prompt_sha256": g1c["prompt_sha256"],
        "rubric_sha256": g1c["rubric_sha256"],
        "context_policy_sha256": _canonical_hash(g1c),
        "metrics": ["assignment_correctness", "context_candidate_recall", "union_candidate_recall", "high_confidence_wrong_identity", "review", "abstention", "conflict"],
        "terminal_policy": _terminal_policy(),
        "action_vector": actions,
        "private_evidence": private,
        "contains_private_membership": True,
        "contains_private_gold": True,
        "did_freeze_cohort": True,
        "did_freeze_gold_commitment": True,
        "did_reveal_gold_to_prediction_workers": False,
        "did_load_or_run_models": False,
    }
    return {**core, "content_sha256": _canonical_hash(core)}


def _portable(preview: Mapping[str, Any]) -> dict[str, Any]:
    hidden = {"private_evidence", "repository_authority"}
    result = {key: value for key, value in preview.items() if key not in hidden}
    result["schema_version"] = RECEIPT_SCHEMA
    result["contains_private_membership"] = False
    result["contains_private_gold"] = False
    return result


def _paths(runtime_root: Path, content_sha256: str) -> dict[str, Path]:
    root = runtime_root.expanduser().absolute()
    run = root / f"generation4-g2-{content_sha256[:24]}"
    return {"root": root, "run": run, "manifest": run / "private-manifest.json", "receipt": run / "receipt.json"}


def apply_generation4_freeze(
    reviewed_preview: Mapping[str, Any], *, expected_content_sha256: str,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
) -> dict[str, Any]:
    preview = preview_generation4_freeze()
    if dict(reviewed_preview) != preview or preview["content_sha256"] != expected_content_sha256:
        raise Generation4FreezeError("Reviewed G2 freeze preview is stale.")
    paths = _paths(runtime_root, expected_content_sha256)
    if paths["manifest"].exists() or paths["receipt"].exists():
        return replay_generation4_freeze(expected_content_sha256, runtime_root=runtime_root)
    ensure_private_tree(paths["root"], paths["run"])
    manifest = {"schema_version": MANIFEST_SCHEMA, "status": "frozen", "preview": preview}
    write_immutable_private_json(paths["manifest"], manifest)
    receipt = {**_portable(preview), "manifest_sha256": sha256_file(paths["manifest"]), "mode": "0600"}
    write_immutable_private_json(paths["receipt"], receipt)
    return {**receipt, "idempotent_replay": False}


def replay_generation4_freeze(
    expected_content_sha256: str, *, runtime_root: Path = DEFAULT_RUNTIME_ROOT
) -> dict[str, Any]:
    preview = preview_generation4_freeze()
    if preview["content_sha256"] != expected_content_sha256:
        raise Generation4FreezeError("Frozen G2 authority drifted.")
    paths = _paths(runtime_root, expected_content_sha256)
    require_private_file(paths["manifest"], paths["root"])
    require_private_file(paths["receipt"], paths["root"])
    manifest = _read_json(paths["manifest"])
    receipt = _read_json(paths["receipt"])
    expected_manifest = {"schema_version": MANIFEST_SCHEMA, "status": "frozen", "preview": preview}
    expected_receipt = {**_portable(preview), "manifest_sha256": sha256_file(paths["manifest"]), "mode": "0600"}
    if manifest != expected_manifest or receipt != expected_receipt:
        raise Generation4FreezeError("Frozen G2 body or receipt drifted.")
    return {**receipt, "replay_schema_version": REPLAY_SCHEMA, "idempotent_replay": True}
