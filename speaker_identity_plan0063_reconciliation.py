"""Build private reviewed-person proposals from exact Plan 0062 human gold."""

from __future__ import annotations

import argparse
import re
import subprocess
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Mapping

from acoustic_audio_derivatives import (
    canonical_artifact_hash,
    ensure_private_tree,
    read_private_object,
    require_private_file,
    sha256_file,
    write_immutable_private_json,
)
import speaker_identity_plan0062_human_comparison as plan0062
from participant_identity import normalize_email
from routing_artifacts import normalize_string


ACTIVATION_SCHEMA = "transcribe-audio.plan0063-a0-activation.v1"
ACTIVATION_RECEIPT_SCHEMA = "transcribe-audio.plan0063-a0-activation-receipt.v1"
RECONCILIATION_SCHEMA = "transcribe-audio.plan0063-person-reconciliation.v1"
RECONCILIATION_RECEIPT_SCHEMA = (
    "transcribe-audio.plan0063-person-reconciliation-receipt.v1"
)
PLAN0063_ACTIVATION_CONTENT_SHA256 = (
    "3c84d2eff1469509184dacf9bbcd163a51953100e3396a7f1a54a8bf614a0139"
)
PLAN0062_SUBMISSION_SHA256 = (
    "5c2ca66fbc25689da8838b65d587fb7f3a5be778a2579f756b8f91526756cdea"
)
PLAN0062_COMPARISON_SHA256 = (
    "372cc17d31c16cdaa4deda47dd8c9fe7cbb057e62f1c6802395fc1dba8d7c84f"
)
PLAN0062_TERMINAL_MANIFEST_SHA256 = (
    "971c5896eaa595069f0387b5f48e5765c1d83e457478f018380ab30534e1f49c"
)
PLAN0062_ENROLLED_BINDING_SHA256 = (
    "79e34705d27608b53776518e8bfe48d3df16a82f139afb13f9236086df3c3c1d"
)
DEFAULT_RUNTIME_ROOT = Path.home() / ".local/state/transcribe-audio/plan-0063"
DEFAULT_PLAN0062_ROOT = Path.home() / ".local/state/transcribe-audio/plan-0062"
SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
ROLE_PLACEHOLDER_TYPES = {"contextual_role_placeholder", "unresolved"}
NAMED_DECISION_TYPES = {
    "canonical_context_proposal",
    "enrolled_voice_subject",
    "contextual_unlisted_suggestion",
    "new_person",
    "corrected_contextual_suggestion",
    "linked_enrolled_context_identity",
}
NEGATIVE_ACTIONS = {
    "apply_speaker_assignments": False,
    "create_biometric_references": False,
    "create_or_update_contacts": False,
    "create_or_update_live_people": False,
    "materialize_biometric_profiles": False,
    "materialize_embeddings": False,
    "reprocess_history": False,
    "restart_watchers": False,
    "write_conversation_observations": False,
    "write_graphiti": False,
    "write_provider_records": False,
}


class Plan0063ReconciliationError(ValueError):
    """Raised when private reviewed-person evidence is incomplete or drifts."""


def _fail(message: str) -> None:
    raise Plan0063ReconciliationError(message)


def _normalized_name(value: Any) -> str:
    return " ".join(
        unicodedata.normalize("NFKC", normalize_string(value)).split()
    ).casefold()


def _stable_id(prefix: str, source_sha256: str, values: Iterable[str]) -> str:
    body = {
        "source_sha256": source_sha256,
        "values": list(values),
    }
    return f"{prefix}-{canonical_artifact_hash(body)[:24]}"


def _git(repo_root: Path, arguments: list[str]) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        _fail("Repository authority could not be read.")
    return result.stdout.strip()


def _repository_authority() -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parent
    if _git(repo_root, ["status", "--porcelain=v1", "--untracked-files=normal"]):
        _fail("Repository must be clean before reconciliation freeze.")
    if _git(
        repo_root,
        ["rev-list", "--left-right", "--count", "HEAD...@{upstream}"],
    ).split() != ["0", "0"]:
        _fail("Repository must be upstream-even before reconciliation freeze.")
    module_paths = (
        repo_root / "speaker_identity_preprocess.py",
        repo_root / "speaker_identity_plan0063_reconciliation.py",
    )
    return {
        "commit": _git(repo_root, ["rev-parse", "HEAD"]),
        "upstream": _git(repo_root, ["rev-parse", "@{upstream}"]),
        "modules": {
            path.name: sha256_file(path)
            for path in module_paths
        },
    }


def _decision_person(decision: Mapping[str, Any]) -> dict[str, str]:
    decision_type = normalize_string(decision.get("decision_type"))
    suggestion = (
        decision.get("suggestion")
        if isinstance(decision.get("suggestion"), Mapping)
        else {}
    )
    name = normalize_string(decision.get("label"))
    if decision_type in {
        "contextual_unlisted_suggestion",
        "linked_enrolled_context_identity",
        "canonical_context_proposal",
    }:
        name = normalize_string(suggestion.get("name")) or name
    return {
        "name": name,
        "email": normalize_email(suggestion.get("email")),
        "organization": normalize_string(suggestion.get("organization")),
    }


def _validated_submission(submission: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    core = {key: value for key, value in submission.items() if key != "content_sha256"}
    decisions = submission.get("decisions")
    if (
        submission.get("schema_version") != plan0062.DECISION_SCHEMA
        or submission.get("status") != "human_gold_frozen_pending_comparison"
        or submission.get("content_sha256") != canonical_artifact_hash(core)
        or not isinstance(decisions, list)
        or int(submission.get("decision_count") or 0) != len(decisions)
        or any((submission.get("negative_actions") or {}).values())
    ):
        _fail("The Plan 0062 human-gold source is invalid.")
    slots: set[str] = set()
    validated: list[Mapping[str, Any]] = []
    for decision in decisions:
        if not isinstance(decision, Mapping):
            _fail("A reviewed speaker decision is not an object.")
        slot_id = normalize_string(decision.get("slot_id"))
        decision_type = normalize_string(decision.get("decision_type"))
        if (
            not slot_id
            or slot_id in slots
            or decision_type not in NAMED_DECISION_TYPES | ROLE_PLACEHOLDER_TYPES
        ):
            _fail("A reviewed speaker decision has an invalid slot or type.")
        if decision_type in NAMED_DECISION_TYPES and not _decision_person(decision)["name"]:
            _fail("A named reviewed speaker decision lacks a person name.")
        slots.add(slot_id)
        validated.append(decision)
    return validated


def _validated_bindings(
    source: Mapping[str, Any], decisions: Iterable[Mapping[str, Any]]
) -> dict[str, Mapping[str, Any]]:
    core = {key: value for key, value in source.items() if key != "content_sha256"}
    bindings = source.get("bindings")
    if (
        source.get("schema_version") != plan0062.BINDING_SCHEMA
        or source.get("status") != "private_enrolled_option_bindings_ready"
        or source.get("content_sha256") != canonical_artifact_hash(core)
        or not isinstance(bindings, list)
        or int(source.get("binding_count") or 0) != len(bindings)
        or any((source.get("negative_actions") or {}).values())
    ):
        _fail("The enrolled-voice binding source is invalid.")
    by_slot: dict[str, Mapping[str, Any]] = {}
    for binding in bindings:
        if not isinstance(binding, Mapping):
            _fail("An enrolled-voice binding is not an object.")
        slot_id = normalize_string(binding.get("slot_id"))
        if slot_id in by_slot or not normalize_string(binding.get("acoustic_subject_id")):
            _fail("An enrolled-voice binding has an invalid slot or subject.")
        by_slot[slot_id] = binding
    decision_by_slot = {
        normalize_string(item.get("slot_id")): item for item in decisions
    }
    for slot_id, binding in by_slot.items():
        decision = decision_by_slot.get(slot_id)
        if not decision or any(
            normalize_string(decision.get(field))
            != normalize_string(binding.get(field))
            for field in (
                "acoustic_subject_id",
                "acoustic_bundle_id",
                "acoustic_bundle_sha256",
            )
        ):
            _fail("A reviewed enrolled-voice decision lost its exact source binding.")
    return by_slot


def build_reconciliation_manifest(
    submission: Mapping[str, Any],
    enrolled_binding_source: Mapping[str, Any],
    *,
    activation_content_sha256: str,
    repository_authority: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create six reviewable person proposals without making any merge authoritative."""

    if not SHA256_RE.fullmatch(activation_content_sha256):
        _fail("The Plan 0063 activation content hash is invalid.")
    decisions = _validated_submission(submission)
    bindings = _validated_bindings(enrolled_binding_source, decisions)
    source_sha256 = normalize_string(submission.get("content_sha256"))

    role_placeholders: list[dict[str, Any]] = []
    slot_identities: list[dict[str, Any]] = []
    for decision in decisions:
        slot_id = normalize_string(decision.get("slot_id"))
        decision_type = normalize_string(decision.get("decision_type"))
        if decision_type in ROLE_PLACEHOLDER_TYPES:
            role_placeholders.append(
                {
                    "slot_id": slot_id,
                    "decision_type": decision_type,
                    "label": normalize_string(decision.get("label")),
                    "creates_person": False,
                }
            )
            continue
        person = _decision_person(decision)
        binding = bindings.get(slot_id) or {}
        slot_identities.append(
            {
                "slot_id": slot_id,
                "slot_person_id": _stable_id(
                    "reviewed-slot-person", source_sha256, [slot_id]
                ),
                "decision_type": decision_type,
                "name": person["name"],
                "normalized_name": _normalized_name(person["name"]),
                "email": person["email"],
                "organization": person["organization"],
                "selected_token": normalize_string(decision.get("selected_token")),
                "acoustic_subject_id": normalize_string(
                    binding.get("acoustic_subject_id")
                ),
                "acoustic_bundle_id": normalize_string(
                    binding.get("acoustic_bundle_id")
                ),
                "acoustic_bundle_sha256": normalize_string(
                    binding.get("acoustic_bundle_sha256")
                ),
            }
        )

    grouped_slots: set[str] = set()
    candidate_groups: list[tuple[str, str, list[dict[str, Any]]]] = []
    by_email: dict[str, list[dict[str, Any]]] = {}
    for item in slot_identities:
        if item["email"]:
            by_email.setdefault(item["email"], []).append(item)
    for email, members in by_email.items():
        if len(members) < 2:
            continue
        candidate_groups.append(("exact_external_identity", email, members))
        grouped_slots.update(item["slot_id"] for item in members)

    by_name: dict[str, list[dict[str, Any]]] = {}
    for item in slot_identities:
        if item["slot_id"] not in grouped_slots:
            by_name.setdefault(item["normalized_name"], []).append(item)
    for name, members in by_name.items():
        basis = "name_only" if len(members) > 1 else "single_reviewed_slot"
        candidate_groups.append((basis, name, members))

    order = {item["slot_id"]: index for index, item in enumerate(slot_identities)}
    candidate_groups.sort(key=lambda group: min(order[item["slot_id"]] for item in group[2]))
    person_proposals: list[dict[str, Any]] = []
    slot_to_proposal: dict[str, str] = {}
    for basis, grouping_value, members in candidate_groups:
        slot_ids = [item["slot_id"] for item in members]
        proposed_person_id = _stable_id(
            "provisional-person", source_sha256, [basis, *sorted(slot_ids)]
        )
        proposal = {
            "proposed_person_id": proposed_person_id,
            "basis": basis,
            "basis_value": grouping_value,
            "member_slot_ids": slot_ids,
            "member_slot_person_ids": [item["slot_person_id"] for item in members],
            "member_names": list(dict.fromkeys(item["name"] for item in members)),
            "external_identities": [
                {"kind": "email", "value": value}
                for value in dict.fromkeys(
                    item["email"] for item in members if item["email"]
                )
            ],
            "requires_human_review": True,
            "authoritative_person_created": False,
            "merge_status": "pending" if len(members) > 1 else "not_applicable",
        }
        person_proposals.append(proposal)
        slot_to_proposal.update({slot_id: proposed_person_id for slot_id in slot_ids})

    merge_proposals = [
        {
            "merge_proposal_id": _stable_id(
                "person-merge",
                source_sha256,
                [proposal["basis"], *proposal["member_slot_ids"]],
            ),
            "proposed_person_id": proposal["proposed_person_id"],
            "basis": proposal["basis"],
            "member_slot_ids": proposal["member_slot_ids"],
            "decision": "pending",
            "authoritative_merge": False,
        }
        for proposal in person_proposals
        if len(proposal["member_slot_ids"]) > 1
    ]
    voice_bindings = [
        {
            "binding_proposal_id": _stable_id(
                "voice-person-binding",
                source_sha256,
                [item["slot_id"], item["acoustic_subject_id"]],
            ),
            "slot_id": item["slot_id"],
            "proposed_person_id": slot_to_proposal[item["slot_id"]],
            "acoustic_subject_id": item["acoustic_subject_id"],
            "acoustic_bundle_id": item["acoustic_bundle_id"],
            "acoustic_bundle_sha256": item["acoustic_bundle_sha256"],
            "context_external_identity": (
                {"kind": "email", "value": item["email"]}
                if item["email"]
                else None
            ),
            "decision": "pending",
            "binding_applied": False,
        }
        for item in slot_identities
        if item["acoustic_subject_id"]
    ]
    voice_person_ids = {item["proposed_person_id"] for item in voice_bindings}
    enrollment_candidates = [
        {
            "proposed_person_id": proposal["proposed_person_id"],
            "member_slot_ids": proposal["member_slot_ids"],
            "eligibility_status": "pending_source_qualification",
            "enrollment_authorized": False,
        }
        for proposal in person_proposals
        if proposal["proposed_person_id"] not in voice_person_ids
    ]
    metrics = {
        "speaker_slot_count": len(decisions),
        "named_slot_count": len(slot_identities),
        "role_placeholder_count": len(role_placeholders),
        "slot_person_identity_count": len(slot_identities),
        "person_proposal_count": len(person_proposals),
        "merge_proposal_count": len(merge_proposals),
        "exact_external_identity_merge_proposal_count": sum(
            item["basis"] == "exact_external_identity" for item in merge_proposals
        ),
        "name_only_merge_proposal_count": sum(
            item["basis"] == "name_only" for item in merge_proposals
        ),
        "existing_voice_binding_proposal_count": len(voice_bindings),
        "new_enrollment_candidate_count": len(enrollment_candidates),
    }
    core = {
        "schema_version": RECONCILIATION_SCHEMA,
        "status": "pending_human_grouping_and_binding_review",
        "activation_content_sha256": activation_content_sha256,
        "source_submission_sha256": source_sha256,
        "source_enrolled_binding_sha256": normalize_string(
            enrolled_binding_source.get("content_sha256")
        ),
        "repository_authority": dict(repository_authority or {}),
        "metrics": metrics,
        "slot_identities": slot_identities,
        "role_placeholders": role_placeholders,
        "person_proposals": person_proposals,
        "merge_proposals": merge_proposals,
        "voice_binding_proposals": voice_bindings,
        "enrollment_candidates": enrollment_candidates,
        "live_mutation_count": 0,
        "negative_actions": dict(NEGATIVE_ACTIONS),
    }
    return {**core, "content_sha256": canonical_artifact_hash(core)}


def _activation_paths(runtime_root: Path, content_sha256: str) -> dict[str, Path]:
    root = runtime_root.expanduser().absolute()
    run = root / f"a0-activation-{content_sha256[:24]}"
    return {
        "root": root,
        "run": run,
        "manifest": run / "private-manifest.json",
        "receipt": run / "receipt.json",
    }


def _reconciliation_paths(runtime_root: Path, content_sha256: str) -> dict[str, Path]:
    root = runtime_root.expanduser().absolute()
    run = root / f"p2-reconciliation-{content_sha256[:24]}"
    return {
        "root": root,
        "run": run,
        "manifest": run / "private-manifest.json",
        "receipt": run / "receipt.json",
    }


def _replay_activation(runtime_root: Path, content_sha256: str) -> dict[str, Any]:
    paths = _activation_paths(runtime_root, content_sha256)
    require_private_file(paths["manifest"], paths["root"])
    require_private_file(paths["receipt"], paths["root"])
    manifest = read_private_object(paths["manifest"])
    receipt = read_private_object(paths["receipt"])
    core = {key: value for key, value in manifest.items() if key != "content_sha256"}
    if (
        manifest.get("schema_version") != ACTIVATION_SCHEMA
        or manifest.get("status") != "open_non_applying"
        or manifest.get("content_sha256") != canonical_artifact_hash(core)
        or manifest.get("content_sha256") != content_sha256
        or any((manifest.get("negative_actions") or {}).values())
        or receipt.get("schema_version") != ACTIVATION_RECEIPT_SCHEMA
        or receipt.get("content_sha256") != content_sha256
        or receipt.get("manifest_sha256") != sha256_file(paths["manifest"])
        or receipt.get("live_mutation_count") != 0
        or receipt.get("a1_required") is not True
        or receipt.get("negative_actions_preserved") is not True
    ):
        _fail("The Plan 0063 A0 activation authority drifted.")
    return manifest


def _exact_sources(plan0062_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = plan0062._paths(plan0062_root, PLAN0062_SUBMISSION_SHA256)
    for key in (
        "decision",
        "decision_receipt",
        "comparison",
        "comparison_receipt",
        "terminal",
    ):
        require_private_file(paths[key], paths["root"])
    submission = read_private_object(paths["decision"])
    comparison = read_private_object(paths["comparison"])
    decision_receipt = read_private_object(paths["decision_receipt"])
    terminal = read_private_object(paths["terminal"])
    if (
        submission.get("content_sha256") != PLAN0062_SUBMISSION_SHA256
        or comparison.get("content_sha256") != PLAN0062_COMPARISON_SHA256
        or sha256_file(paths["terminal"]) != PLAN0062_TERMINAL_MANIFEST_SHA256
        or decision_receipt.get("decision_manifest_sha256")
        != sha256_file(paths["decision"])
        or decision_receipt.get("live_mutation_count") != 0
        or terminal.get("submission_content_sha256") != PLAN0062_SUBMISSION_SHA256
        or terminal.get("comparison_content_sha256") != PLAN0062_COMPARISON_SHA256
        or terminal.get("live_mutation_count") != 0
        or any((terminal.get("negative_actions") or {}).values())
    ):
        _fail("The exact Plan 0062 P5 source authority drifted.")
    binding_receipt = plan0062.replay_enrolled_option_bindings(
        content_sha256=PLAN0062_ENROLLED_BINDING_SHA256,
        runtime_root=plan0062_root,
    )
    binding_source = read_private_object(Path(binding_receipt["manifest_path"]))
    if (
        submission.get("enrolled_binding_content_sha256")
        != PLAN0062_ENROLLED_BINDING_SHA256
    ):
        _fail("The Plan 0062 submission lost its enrolled-binding authority.")
    return submission, binding_source


def freeze_exact_reconciliation(
    *,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    plan0062_root: Path = DEFAULT_PLAN0062_ROOT,
) -> dict[str, Any]:
    """Freeze the exact non-applying Plan 0063 P2 private reconciliation."""

    _replay_activation(runtime_root, PLAN0063_ACTIVATION_CONTENT_SHA256)
    submission, binding_source = _exact_sources(plan0062_root)
    manifest = build_reconciliation_manifest(
        submission,
        binding_source,
        activation_content_sha256=PLAN0063_ACTIVATION_CONTENT_SHA256,
        repository_authority=_repository_authority(),
    )
    expected_metrics = {
        "speaker_slot_count": 10,
        "named_slot_count": 9,
        "role_placeholder_count": 1,
        "slot_person_identity_count": 9,
        "person_proposal_count": 6,
        "merge_proposal_count": 3,
        "exact_external_identity_merge_proposal_count": 1,
        "name_only_merge_proposal_count": 2,
        "existing_voice_binding_proposal_count": 1,
        "new_enrollment_candidate_count": 5,
    }
    if manifest.get("metrics") != expected_metrics:
        _fail("The exact Plan 0063 reconciliation denominator drifted.")
    paths = _reconciliation_paths(runtime_root, manifest["content_sha256"])
    if paths["receipt"].exists():
        return replay_reconciliation(
            content_sha256=manifest["content_sha256"], runtime_root=runtime_root
        )
    if paths["run"].exists():
        _fail("A partial Plan 0063 reconciliation directory already exists.")
    ensure_private_tree(paths["root"], paths["run"])
    write_immutable_private_json(paths["manifest"], manifest)
    receipt = {
        "schema_version": RECONCILIATION_RECEIPT_SCHEMA,
        "status": "private_reconciliation_frozen_pending_review",
        "content_sha256": manifest["content_sha256"],
        "manifest_sha256": sha256_file(paths["manifest"]),
        "metrics": expected_metrics,
        "live_mutation_count": 0,
        "negative_actions_preserved": True,
    }
    write_immutable_private_json(paths["receipt"], receipt)
    return {
        **receipt,
        "manifest_path": str(paths["manifest"]),
        "receipt_path": str(paths["receipt"]),
        "idempotent_replay": False,
    }


def replay_reconciliation(
    *, content_sha256: str, runtime_root: Path = DEFAULT_RUNTIME_ROOT
) -> dict[str, Any]:
    """Replay an immutable Plan 0063 P2 reconciliation without exposing values."""

    if not SHA256_RE.fullmatch(content_sha256):
        _fail("The Plan 0063 reconciliation content hash is invalid.")
    paths = _reconciliation_paths(runtime_root, content_sha256)
    require_private_file(paths["manifest"], paths["root"])
    require_private_file(paths["receipt"], paths["root"])
    manifest = read_private_object(paths["manifest"])
    receipt = read_private_object(paths["receipt"])
    core = {key: value for key, value in manifest.items() if key != "content_sha256"}
    if (
        manifest.get("schema_version") != RECONCILIATION_SCHEMA
        or manifest.get("status") != "pending_human_grouping_and_binding_review"
        or manifest.get("content_sha256") != content_sha256
        or canonical_artifact_hash(core) != content_sha256
        or any((manifest.get("negative_actions") or {}).values())
        or receipt.get("schema_version") != RECONCILIATION_RECEIPT_SCHEMA
        or receipt.get("content_sha256") != content_sha256
        or receipt.get("manifest_sha256") != sha256_file(paths["manifest"])
        or receipt.get("metrics") != manifest.get("metrics")
        or receipt.get("live_mutation_count") != 0
        or receipt.get("negative_actions_preserved") is not True
    ):
        _fail("The frozen Plan 0063 reconciliation drifted.")
    return {
        **receipt,
        "manifest_path": str(paths["manifest"]),
        "receipt_path": str(paths["receipt"]),
        "idempotent_replay": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "replay"))
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--plan0062-root", type=Path, default=DEFAULT_PLAN0062_ROOT)
    parser.add_argument("--content-sha256", default="")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "freeze":
        result = freeze_exact_reconciliation(
            runtime_root=args.runtime_root,
            plan0062_root=args.plan0062_root,
        )
    else:
        result = replay_reconciliation(
            content_sha256=args.content_sha256,
            runtime_root=args.runtime_root,
        )
    print(
        {
            "status": result["status"],
            "content_sha256": result["content_sha256"],
            "manifest_sha256": result["manifest_sha256"],
            "metrics": result["metrics"],
            "live_mutation_count": result["live_mutation_count"],
            "idempotent_replay": result["idempotent_replay"],
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
