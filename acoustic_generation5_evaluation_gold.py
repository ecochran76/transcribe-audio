"""Build Plan 0054 E1 private gold and deterministic cohort feasibility."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import unicodedata
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import acoustic_generation5_evaluation_authority as e1
from acoustic_audio_derivatives import (
    ensure_private_tree,
    require_private_file,
    sha256_file,
    write_immutable_private_json,
)


PREVIEW_SCHEMA = "transcribe-audio.generation5-e1-gold-preview.v1"
MANIFEST_SCHEMA = "transcribe-audio.generation5-e1-gold-manifest.v1"
RECEIPT_SCHEMA = "transcribe-audio.generation5-e1-gold-receipt.v1"
REPLAY_SCHEMA = "transcribe-audio.generation5-e1-gold-replay.v1"
DEFAULT_RUNTIME_ROOT = Path("~/.local/state/transcribe-audio/plan-0054/e1-gold")
E1_PREVIEW_SHA256 = "eab13de1ac89ffafef8dca228368ddbe792341b9796d4a0f503ce3d5405dd6e1"
E1_MANIFEST_SHA256 = "63e62c3199c6e2c02688ff365fc29c9926f40202fe03c8c49b1a2646912f2700"
GENERATION3_GOLD_MANIFEST = Path(
    "~/.local/state/transcribe-audio/plan-0037/generation-3/gold-authorities/"
    "generation3-gold-5f60fa794c40c8fa5a2c5cb0/private-manifest.json"
)
GENERATION3_GOLD_MANIFEST_SHA256 = "5e91c62985d137ca64689e6cd49872b92ebce1051689d62f43e32d000824495e"
GENERATION3_GOLD_MANIFEST_SCHEMA = "transcribe-audio.generation3-gold-manifest.v1"
MODULE_NAME = Path(__file__).name
CASE_RE = re.compile(r"g5-case-[a-f0-9]{20}")


class Generation5EvaluationGoldError(ValueError):
    """Raised when private E1 gold or population evidence is invalid."""


def _canonical_hash(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _normalized_identity(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return " ".join(text.split()).casefold()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Generation5EvaluationGoldError("Private gold authority is unreadable.") from exc
    if not isinstance(value, dict):
        raise Generation5EvaluationGoldError("Private gold authority must be an object.")
    return value


def _git(arguments: Sequence[str], *, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", *arguments],
        cwd=Path(__file__).resolve().parent,
        capture_output=True,
        text=not binary,
        check=False,
    )
    if result.returncode:
        raise Generation5EvaluationGoldError("Repository authority is unavailable.")
    return result.stdout if binary else result.stdout.strip()


def _repository_authority() -> dict[str, Any]:
    if _git(["status", "--porcelain=v1", "--untracked-files=normal"]):
        raise Generation5EvaluationGoldError("Repository must be clean.")
    parity = str(_git(["rev-list", "--left-right", "--count", "@{upstream}...HEAD"])).split()
    if parity != ["0", "0"]:
        raise Generation5EvaluationGoldError("Repository must be upstream-even.")
    commit = str(_git(["rev-parse", "HEAD"]))
    body = _git(["show", f"{commit}:{MODULE_NAME}"], binary=True)
    if (
        not re.fullmatch(r"[a-f0-9]{40}", commit)
        or not isinstance(body, bytes)
        or hashlib.sha256(body).hexdigest() != sha256_file(Path(__file__).resolve())
    ):
        raise Generation5EvaluationGoldError("Committed E1 gold module drifted.")
    return {
        "commit": commit,
        "module_sha256": hashlib.sha256(body).hexdigest(),
        "clean": True,
        "upstream_ahead": 0,
        "upstream_behind": 0,
    }


def _e1_preview() -> dict[str, Any]:
    replay = e1.replay_generation5_evaluation_authority(E1_PREVIEW_SHA256)
    paths = e1._paths(e1.DEFAULT_RUNTIME_ROOT, E1_PREVIEW_SHA256)
    if replay.get("idempotent_replay") is not True or sha256_file(paths["manifest"]) != E1_MANIFEST_SHA256:
        raise Generation5EvaluationGoldError("E1 review authority drifted.")
    manifest = _read_json(paths["manifest"])
    preview = manifest.get("preview")
    if (
        not isinstance(preview, Mapping)
        or preview.get("content_sha256") != E1_PREVIEW_SHA256
        or preview.get("candidate_count") != 11
        or preview.get("speaker_label_count") != 29
        or preview.get("did_access_identity_gold") is not False
        or preview.get("did_load_or_run_models") is not False
    ):
        raise Generation5EvaluationGoldError("E1 review denominator drifted.")
    return dict(preview)


def _enrolled_identity_map() -> dict[str, str]:
    path = GENERATION3_GOLD_MANIFEST.expanduser().absolute()
    require_private_file(path, path.parents[1])
    manifest = _read_json(path)
    preview = manifest.get("preview")
    bindings = preview.get("enrolled_identity_bindings") if isinstance(preview, Mapping) else None
    if (
        sha256_file(path) != GENERATION3_GOLD_MANIFEST_SHA256
        or manifest.get("schema_version") != GENERATION3_GOLD_MANIFEST_SCHEMA
        or manifest.get("status") != "applied_gold_frozen_evaluation_not_revealed"
        or not isinstance(bindings, list)
        or len(bindings) != 2
    ):
        raise Generation5EvaluationGoldError("Enrolled identity authority drifted.")
    result = {}
    for raw in bindings:
        if not isinstance(raw, Mapping):
            raise Generation5EvaluationGoldError("Enrolled identity binding is invalid.")
        identity = " ".join(str(raw.get("identity_name") or "").split())
        normalized = _normalized_identity(identity)
        subject = str(raw.get("person_ref_id") or "")
        if (
            not normalized
            or hashlib.sha256(normalized.encode()).hexdigest() != raw.get("identity_name_sha256")
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{2,127}", subject)
            or normalized in result
        ):
            raise Generation5EvaluationGoldError("Enrolled identity binding is invalid.")
        result[normalized] = subject
    return result


def parse_review_answers(answer_text: str, *, expected_refs: Sequence[str]) -> dict[str, str]:
    """Parse the copyable answer block and reject partial or uncertain gold."""
    expected = list(expected_refs)
    if not expected or len(expected) != len(set(expected)):
        raise Generation5EvaluationGoldError("Expected speaker references are invalid.")
    answers = {}
    ignored_page_lines = {
        "private generation-5 speaker-label review",
        "prepare answers",
        "transcript clues",
        "identity or stable alias",
        "reuse the same name for the same person. use unknown only if you cannot tell.",
    }
    for raw_line in str(answer_text or "").splitlines():
        line = raw_line.strip()
        if not line or line.casefold() in ignored_page_lines:
            continue
        if "=" not in line:
            if line in expected or line.startswith("These are the 12 fresh candidates"):
                continue
            raise Generation5EvaluationGoldError("Every answer line must contain '='.")
        reference, raw_value = line.split("=", 1)
        reference = reference.strip()
        value = " ".join(raw_value.split())
        if reference not in expected or reference in answers:
            raise Generation5EvaluationGoldError("Answer references are unknown or duplicated.")
        if (
            not value
            or _normalized_identity(value) in {"unknown", "unanswered", "not sure", "unsure"}
            or len(value) > 200
            or any(ord(character) < 32 for character in value)
        ):
            raise Generation5EvaluationGoldError("Every speaker needs a stable identity or alias.")
        answers[reference] = value
    if set(answers) != set(expected):
        raise Generation5EvaluationGoldError("The private speaker review is incomplete.")
    return answers


def _person_id(identity: str) -> str:
    return "g5-person-" + hashlib.sha256(_normalized_identity(identity).encode()).hexdigest()[:24]


def _conversation_id(case_id: str) -> str:
    return "g5-conversation-" + hashlib.sha256(case_id.encode()).hexdigest()[:24]


def _recording_id(source_sha256: str) -> str:
    return "g5-recording-" + hashlib.sha256(source_sha256.encode()).hexdigest()[:24]


def _cases(preview: Mapping[str, Any], answers: Mapping[str, str], enrolled: Mapping[str, str]) -> list[dict[str, Any]]:
    private = preview.get("private_evidence")
    cards = private.get("cards") if isinstance(private, Mapping) else None
    membership = private.get("candidate_membership") if isinstance(private, Mapping) else None
    if not isinstance(cards, list) or not isinstance(membership, list):
        raise Generation5EvaluationGoldError("E1 private review membership is unavailable.")
    ordinal_by_pair = {
        (str(row.get("source_sha256") or ""), str(row.get("transcript_sha256") or "")): int(row.get("enumerated_ordinal") or 0)
        for row in membership
        if isinstance(row, Mapping)
    }
    cases_by_id = {}
    for raw in cards:
        if not isinstance(raw, Mapping):
            raise Generation5EvaluationGoldError("E1 private review card is invalid.")
        card = dict(raw)
        case_id = str(card.get("case_id") or "")
        reference = str(card.get("speaker_ref") or "")
        label = str(card.get("speaker_label") or "")
        source_hash = str(card.get("source_sha256") or "")
        transcript_hash = str(card.get("transcript_sha256") or "")
        expected_case = "g5-case-" + _canonical_hash({"source": source_hash, "transcript": transcript_hash})[:20]
        ordinal = ordinal_by_pair.get((source_hash, transcript_hash), 0)
        if (
            not CASE_RE.fullmatch(case_id)
            or case_id != expected_case
            or not reference
            or reference not in answers
            or not label
            or ordinal <= 0
        ):
            raise Generation5EvaluationGoldError("E1 private review membership drifted.")
        identity = answers[reference]
        case = cases_by_id.setdefault(
            case_id,
            {
                "case_id": case_id,
                "enumerated_ordinal": ordinal,
                "source_sha256": source_hash,
                "transcript_sha256": transcript_hash,
                "conversation_id": _conversation_id(case_id),
                "recording_id": _recording_id(source_hash),
                "speaker_gold": [],
                "overlap_codes": [],
            },
        )
        if case["source_sha256"] != source_hash or case["transcript_sha256"] != transcript_hash:
            raise Generation5EvaluationGoldError("An E1 case has conflicting membership.")
        case["speaker_gold"].append(
            {
                "speaker_label": label,
                "person_id": _person_id(identity),
                "enrolled_subject_id": enrolled.get(_normalized_identity(identity), ""),
                "private_identity_display": identity,
                "operator_answer_sha256": hashlib.sha256(f"{reference}={identity}".encode()).hexdigest(),
            }
        )
    return sorted(cases_by_id.values(), key=lambda case: case["enumerated_ordinal"])


def evaluate_population(cases: Sequence[Mapping[str, Any]], *, expected_sources: set[str]) -> dict[str, Any]:
    conversations = set()
    recordings = set()
    sources = set()
    transcripts = set()
    people_sessions: dict[str, set[str]] = defaultdict(set)
    enrolled_sessions: dict[str, set[str]] = defaultdict(set)
    overlap_codes = set()
    complete = len(cases) == 7
    for case in cases:
        source = str(case.get("source_sha256") or "")
        transcript = str(case.get("transcript_sha256") or "")
        conversation = str(case.get("conversation_id") or "")
        recording = str(case.get("recording_id") or "")
        speakers = case.get("speaker_gold")
        if not source or not transcript or not conversation or not recording or not isinstance(speakers, list) or not speakers:
            complete = False
            continue
        if source in sources:
            overlap_codes.add("duplicate_source")
        if transcript in transcripts:
            overlap_codes.add("duplicate_derivative")
        if conversation in conversations:
            overlap_codes.add("duplicate_conversation")
        if recording in recordings:
            overlap_codes.add("duplicate_recording")
        sources.add(source)
        transcripts.add(transcript)
        conversations.add(conversation)
        recordings.add(recording)
        for speaker in speakers:
            if not isinstance(speaker, Mapping) or not speaker.get("speaker_label") or not speaker.get("person_id"):
                complete = False
                continue
            people_sessions[str(speaker["person_id"])].add(conversation)
            enrolled_subject = str(speaker.get("enrolled_subject_id") or "")
            if enrolled_subject:
                enrolled_sessions[enrolled_subject].add(conversation)
        overlap_codes.update(str(code) for code in case.get("overlap_codes") or [])
    if sources - expected_sources:
        overlap_codes.add("source_outside_authority")
    same_person_pairs = sum(len(tuple(combinations(sessions, 2))) for sessions in people_sessions.values())
    enrolled_with_two = sum(len(sessions) >= 2 for sessions in enrolled_sessions.values())
    gates = {
        "exactly_seven_conversations": len(conversations) == 7,
        "both_enrolled_people_have_two_recordings": enrolled_with_two == 2,
        "minimum_five_people": len(people_sessions) >= 5,
        "minimum_four_same_person_session_pairs": same_person_pairs >= 4,
        "complete_private_gold": complete,
        "zero_overlap": not overlap_codes,
        "all_sources_within_authority": sources <= expected_sources,
    }
    return {
        "conversation_count": len(conversations),
        "recording_count": len(recordings),
        "person_count": len(people_sessions),
        "enrolled_people_with_two_recordings_count": enrolled_with_two,
        "same_person_session_pair_count": same_person_pairs,
        "complete_gold_case_count": len(cases) if complete else 0,
        "overlap_count": len(overlap_codes),
        "overlap_code_set_sha256": _canonical_hash(sorted(overlap_codes)),
        "gates": gates,
        "passing": all(gates.values()),
    }


def select_first_passing_seven(cases: Sequence[Mapping[str, Any]], *, expected_sources: set[str]) -> tuple[list[dict[str, Any]], dict[str, Any] | None, int]:
    ordered = sorted((dict(case) for case in cases), key=lambda case: int(case.get("enumerated_ordinal") or 0))
    checked = 0
    for subset in combinations(ordered, 7):
        checked += 1
        result = evaluate_population(subset, expected_sources=expected_sources)
        if result["passing"]:
            return list(subset), result, checked
    return [], None, checked


def preview_generation5_evaluation_gold(
    answer_text: str,
    *,
    e1_preview: Mapping[str, Any] | None = None,
    enrolled_identity_map: Mapping[str, str] | None = None,
    repository_authority: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    authority = dict(e1_preview or _e1_preview())
    if authority.get("content_sha256") != E1_PREVIEW_SHA256:
        raise Generation5EvaluationGoldError("E1 review authority is invalid.")
    private = authority.get("private_evidence")
    cards = private.get("cards") if isinstance(private, Mapping) else None
    if not isinstance(cards, list) or len(cards) != authority.get("speaker_label_count"):
        raise Generation5EvaluationGoldError("E1 review cards are incomplete.")
    expected_refs = [str(card.get("speaker_ref") or "") for card in cards if isinstance(card, Mapping)]
    answers = parse_review_answers(answer_text, expected_refs=expected_refs)
    enrolled = dict(enrolled_identity_map or _enrolled_identity_map())
    if len(enrolled) != 2:
        raise Generation5EvaluationGoldError("Exactly two enrolled identities are required.")
    cases = _cases(authority, answers, enrolled)
    expected_sources = {str(case["source_sha256"]) for case in cases}
    selected, population, combinations_checked = select_first_passing_seven(cases, expected_sources=expected_sources)
    passing = bool(selected and population and population.get("passing"))
    answer_projection = sorted(
        ({"speaker_ref": ref, "identity_sha256": hashlib.sha256(_normalized_identity(value).encode()).hexdigest()} for ref, value in answers.items()),
        key=lambda item: item["speaker_ref"],
    )
    selected_ids = [str(case["case_id"]) for case in selected]
    private_gold = {
        "all_cases": cases,
        "selected_cases": selected,
        "selected_case_ids": selected_ids,
    }
    actions = {
        "submit_exact_cohort_and_gold_feasibility_to_j3": passing,
        "freeze_cohort_or_gold": False,
        "run_models_or_predictions": False,
        "reveal_gold_to_workers": False,
        "mutate_profiles_or_references": False,
        "enable_default_integration": False,
        "run_historical_reprocessing": False,
    }
    core = {
        "schema_version": PREVIEW_SCHEMA,
        "status": "ready_for_independent_j3_review" if passing else "population_infeasible_stop",
        "repository_authority": dict(repository_authority or _repository_authority()),
        "e1_preview_sha256": E1_PREVIEW_SHA256,
        "e1_manifest_sha256": E1_MANIFEST_SHA256,
        "generation3_gold_manifest_sha256": GENERATION3_GOLD_MANIFEST_SHA256,
        "answer_set_sha256": _canonical_hash(answer_projection),
        "reviewed_candidate_count": len(cases),
        "reviewed_speaker_label_count": len(cards),
        "combination_size": 7,
        "combinations_checked": combinations_checked,
        "population_feasible": passing,
        "population_result": population or {},
        "selected_case_ids_sha256": _canonical_hash(selected_ids),
        "selected_source_set_sha256": _canonical_hash(sorted(str(case["source_sha256"]) for case in selected)),
        "selected_transcript_set_sha256": _canonical_hash(sorted(str(case["transcript_sha256"]) for case in selected)),
        "private_gold": private_gold,
        "action_vector": actions,
        "contains_paths": False,
        "contains_private_membership": True,
        "contains_identity_names_or_aliases": True,
        "contains_transcript_text": False,
        "contains_audio": False,
        "did_freeze_cohort_or_gold": False,
        "did_load_or_run_models": False,
        "did_reveal_gold_to_workers": False,
    }
    return {**core, "content_sha256": _canonical_hash(core)}


def _portable(preview: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": RECEIPT_SCHEMA,
        "status": preview["status"],
        "preview_content_sha256": preview["content_sha256"],
        "answer_set_sha256": preview["answer_set_sha256"],
        "reviewed_candidate_count": preview["reviewed_candidate_count"],
        "reviewed_speaker_label_count": preview["reviewed_speaker_label_count"],
        "combination_size": preview["combination_size"],
        "combinations_checked": preview["combinations_checked"],
        "population_feasible": preview["population_feasible"],
        "population_result": preview["population_result"],
        "selected_case_ids_sha256": preview["selected_case_ids_sha256"],
        "selected_source_set_sha256": preview["selected_source_set_sha256"],
        "selected_transcript_set_sha256": preview["selected_transcript_set_sha256"],
        "action_vector": preview["action_vector"],
        "did_freeze_cohort_or_gold": False,
        "did_load_or_run_models": False,
        "did_reveal_gold_to_workers": False,
    }


def _validated_proposal(
    raw_preview: Mapping[str, Any], *, require_current_repository: bool
) -> dict[str, Any]:
    """Recompute E1 membership, population, and selection before persistence."""
    preview = dict(raw_preview)
    core = {key: value for key, value in preview.items() if key != "content_sha256"}
    if preview.get("schema_version") != PREVIEW_SCHEMA or preview.get("content_sha256") != _canonical_hash(core):
        raise Generation5EvaluationGoldError("Reviewed E1 gold preview is stale.")
    repository = preview.get("repository_authority")
    if not isinstance(repository, Mapping):
        raise Generation5EvaluationGoldError("E1 gold repository authority is missing.")
    if require_current_repository and dict(repository) != _repository_authority():
        raise Generation5EvaluationGoldError("E1 gold repository authority is stale.")

    authority = _e1_preview()
    authority_private = authority.get("private_evidence")
    membership = authority_private.get("candidate_membership") if isinstance(authority_private, Mapping) else None
    cards = authority_private.get("cards") if isinstance(authority_private, Mapping) else None
    private_gold = preview.get("private_gold")
    all_cases = private_gold.get("all_cases") if isinstance(private_gold, Mapping) else None
    selected_cases = private_gold.get("selected_cases") if isinstance(private_gold, Mapping) else None
    selected_ids = private_gold.get("selected_case_ids") if isinstance(private_gold, Mapping) else None
    if not all(isinstance(value, list) for value in (membership, cards, all_cases, selected_cases, selected_ids)):
        raise Generation5EvaluationGoldError("E1 gold private membership is incomplete.")

    expected_members = [
        (
            int(row.get("enumerated_ordinal") or 0),
            str(row.get("source_sha256") or ""),
            str(row.get("transcript_sha256") or ""),
        )
        for row in membership
        if isinstance(row, Mapping)
    ]
    actual_members = [
        (
            int(case.get("enumerated_ordinal") or 0),
            str(case.get("source_sha256") or ""),
            str(case.get("transcript_sha256") or ""),
        )
        for case in all_cases
        if isinstance(case, Mapping)
    ]
    expected_card_labels = sorted(
        (
            str(card.get("case_id") or ""),
            str(card.get("speaker_label") or ""),
        )
        for card in cards
        if isinstance(card, Mapping)
    )
    actual_card_labels = sorted(
        (str(case.get("case_id") or ""), str(speaker.get("speaker_label") or ""))
        for case in all_cases
        if isinstance(case, Mapping)
        for speaker in case.get("speaker_gold") or []
        if isinstance(speaker, Mapping)
    )
    enrolled_map = _enrolled_identity_map()
    reference_by_label = {
        (str(card.get("case_id") or ""), str(card.get("speaker_label") or "")): str(card.get("speaker_ref") or "")
        for card in cards
        if isinstance(card, Mapping)
    }
    answer_projection = []
    speaker_bindings_valid = True
    for case in all_cases:
        if not isinstance(case, Mapping):
            speaker_bindings_valid = False
            continue
        case_id = str(case.get("case_id") or "")
        for speaker in case.get("speaker_gold") or []:
            if not isinstance(speaker, Mapping):
                speaker_bindings_valid = False
                continue
            label = str(speaker.get("speaker_label") or "")
            reference = reference_by_label.get((case_id, label), "")
            identity = " ".join(str(speaker.get("private_identity_display") or "").split())
            normalized = _normalized_identity(identity)
            if (
                not reference
                or not normalized
                or speaker.get("person_id") != _person_id(identity)
                or speaker.get("enrolled_subject_id") != enrolled_map.get(normalized, "")
                or speaker.get("operator_answer_sha256")
                != hashlib.sha256(f"{reference}={identity}".encode()).hexdigest()
            ):
                speaker_bindings_valid = False
            answer_projection.append(
                {
                    "speaker_ref": reference,
                    "identity_sha256": hashlib.sha256(normalized.encode()).hexdigest(),
                }
            )
    if (
        expected_members != actual_members
        or expected_card_labels != actual_card_labels
        or not speaker_bindings_valid
        or preview.get("answer_set_sha256")
        != _canonical_hash(sorted(answer_projection, key=lambda item: item["speaker_ref"]))
        or preview.get("e1_preview_sha256") != E1_PREVIEW_SHA256
        or preview.get("e1_manifest_sha256") != E1_MANIFEST_SHA256
        or preview.get("generation3_gold_manifest_sha256") != GENERATION3_GOLD_MANIFEST_SHA256
        or preview.get("reviewed_candidate_count") != len(all_cases)
        or preview.get("reviewed_speaker_label_count") != len(cards)
        or preview.get("combination_size") != 7
        or preview.get("did_freeze_cohort_or_gold") is not False
        or preview.get("did_load_or_run_models") is not False
        or preview.get("did_reveal_gold_to_workers") is not False
    ):
        raise Generation5EvaluationGoldError("E1 gold membership drifted.")

    expected_sources = {source for _, source, _ in expected_members}
    recomputed_cases, recomputed_population, recomputed_checked = select_first_passing_seven(
        all_cases, expected_sources=expected_sources
    )
    recomputed_ids = [str(case["case_id"]) for case in recomputed_cases]
    passing = bool(recomputed_cases and recomputed_population and recomputed_population.get("passing"))
    expected_status = "ready_for_independent_j3_review" if passing else "population_infeasible_stop"
    action = preview.get("action_vector")
    if (
        selected_cases != recomputed_cases
        or selected_ids != recomputed_ids
        or preview.get("population_result") != (recomputed_population or {})
        or preview.get("population_feasible") is not passing
        or preview.get("combinations_checked") != recomputed_checked
        or preview.get("status") != expected_status
        or preview.get("selected_case_ids_sha256") != _canonical_hash(recomputed_ids)
        or preview.get("selected_source_set_sha256")
        != _canonical_hash(sorted(str(case["source_sha256"]) for case in recomputed_cases))
        or preview.get("selected_transcript_set_sha256")
        != _canonical_hash(sorted(str(case["transcript_sha256"]) for case in recomputed_cases))
        or not isinstance(action, Mapping)
        or action.get("submit_exact_cohort_and_gold_feasibility_to_j3") is not passing
        or any(
            action.get(key) is not False
            for key in (
                "freeze_cohort_or_gold",
                "run_models_or_predictions",
                "reveal_gold_to_workers",
                "mutate_profiles_or_references",
                "enable_default_integration",
                "run_historical_reprocessing",
            )
        )
    ):
        raise Generation5EvaluationGoldError("E1 population selection drifted.")
    return preview


def _paths(runtime_root: Path, content_sha256: str) -> dict[str, Path]:
    root = runtime_root.expanduser().absolute()
    run = root / f"generation5-e1-gold-{content_sha256[:24]}"
    return {"root": root, "run": run, "manifest": run / "private-manifest.json", "receipt": run / "receipt.json"}


def apply_generation5_evaluation_gold(
    reviewed_preview: Mapping[str, Any],
    *,
    expected_content_sha256: str,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
) -> dict[str, Any]:
    preview = _validated_proposal(reviewed_preview, require_current_repository=True)
    if preview.get("content_sha256") != expected_content_sha256:
        raise Generation5EvaluationGoldError("Reviewed E1 gold preview is stale.")
    paths = _paths(runtime_root, expected_content_sha256)
    if paths["manifest"].exists() or paths["receipt"].exists():
        return replay_generation5_evaluation_gold(expected_content_sha256, runtime_root=runtime_root)
    ensure_private_tree(paths["root"], paths["run"])
    manifest = {"schema_version": MANIFEST_SCHEMA, "status": "frozen_proposal", "preview": preview}
    write_immutable_private_json(paths["manifest"], manifest)
    receipt = {**_portable(preview), "manifest_sha256": sha256_file(paths["manifest"]), "mode": "0600"}
    write_immutable_private_json(paths["receipt"], receipt)
    return {**receipt, "idempotent_replay": False}


def replay_generation5_evaluation_gold(
    expected_content_sha256: str,
    *,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
) -> dict[str, Any]:
    paths = _paths(runtime_root, expected_content_sha256)
    require_private_file(paths["manifest"], paths["root"])
    require_private_file(paths["receipt"], paths["root"])
    manifest = _read_json(paths["manifest"])
    receipt = _read_json(paths["receipt"])
    preview = manifest.get("preview")
    if not isinstance(preview, Mapping):
        raise Generation5EvaluationGoldError("Private E1 gold preview is missing.")
    preview = _validated_proposal(preview, require_current_repository=False)
    repository = preview.get("repository_authority")
    if not isinstance(repository, Mapping):
        raise Generation5EvaluationGoldError("Recorded E1 gold repository authority is missing.")
    commit = str(repository.get("commit") or "")
    body = _git(["show", f"{commit}:{MODULE_NAME}"], binary=True) if re.fullmatch(r"[a-f0-9]{40}", commit) else b""
    _e1_preview()
    core = {key: value for key, value in preview.items() if key != "content_sha256"}
    expected_manifest = {"schema_version": MANIFEST_SCHEMA, "status": "frozen_proposal", "preview": preview}
    expected_receipt = {**_portable(preview), "manifest_sha256": sha256_file(paths["manifest"]), "mode": "0600"}
    if (
        preview.get("content_sha256") != expected_content_sha256
        or _canonical_hash(core) != expected_content_sha256
        or manifest != expected_manifest
        or receipt != expected_receipt
        or not isinstance(body, bytes)
        or hashlib.sha256(body).hexdigest() != repository.get("module_sha256")
        or _git(["merge-base", "--is-ancestor", commit, "HEAD"]) != ""
    ):
        raise Generation5EvaluationGoldError("Private E1 gold authority drifted.")
    return {**receipt, "replay_schema_version": REPLAY_SCHEMA, "idempotent_replay": True}
