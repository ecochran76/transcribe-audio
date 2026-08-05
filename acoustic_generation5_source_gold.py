"""Build Plan 0055 private gold and deterministic population proposal."""

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

import acoustic_generation5_evaluation_gold as prior_gold
import acoustic_generation5_source_review as s1
from acoustic_audio_derivatives import (
    ensure_private_tree,
    require_private_file,
    sha256_file,
    write_immutable_private_json,
)


PREVIEW_SCHEMA = "transcribe-audio.generation5-source-gold-preview.v1"
MANIFEST_SCHEMA = "transcribe-audio.generation5-source-gold-manifest.v1"
RECEIPT_SCHEMA = "transcribe-audio.generation5-source-gold-receipt.v1"
REPLAY_SCHEMA = "transcribe-audio.generation5-source-gold-replay.v1"
S1_PREVIEW_SHA256 = "5a3f9fc9848a5e0b669bc37796e5a55b4f9dcd7bf0f55609aefa886e4caabcf9"
S1_MANIFEST_SHA256 = "04a860f3823c82f513dc655970fbe9b9d99641cec3f343d2e055095f39bb9a84"
DEFAULT_RUNTIME_ROOT = Path("~/.local/state/transcribe-audio/plan-0055/s1/gold-proposal")
MODULE_NAME = Path(__file__).name
CASE_RE = re.compile(r"g5s-case-[a-f0-9]{20}")


class Generation5SourceGoldError(ValueError):
    """Raised when Plan 0055 private gold or population evidence drifts."""


def _canonical_hash(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(body.encode()).hexdigest()


def _normalized_identity(value: Any) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value or "")).split()).casefold()


def _canonical_identity(value: str) -> str:
    display = " ".join(str(value).split())
    aliases = {
        "dr. jeffrey dikis": "Jeffrey Dikis",
        "jeffrey dikis": "Jeffrey Dikis",
        "dr. dikis nurse": "Dr. Dikis' Nurse",
        "dr. dikis' nurse": "Dr. Dikis' Nurse",
        "alexendra hoen": "Alexandra Hoen",
        "alexandra hoen": "Alexandra Hoen",
    }
    return aliases.get(_normalized_identity(display), display)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Generation5SourceGoldError("Private gold evidence is unreadable.") from exc
    if not isinstance(value, dict):
        raise Generation5SourceGoldError("Private gold evidence must be an object.")
    return value


def _git(arguments: Sequence[str], *, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", *arguments], cwd=Path(__file__).resolve().parent,
        capture_output=True, text=not binary, check=False,
    )
    if result.returncode:
        raise Generation5SourceGoldError("Repository authority is unavailable.")
    return result.stdout if binary else result.stdout.strip()


def _repository_authority() -> dict[str, Any]:
    if _git(["status", "--porcelain=v1", "--untracked-files=normal"]):
        raise Generation5SourceGoldError("Repository must be clean.")
    if str(_git(["rev-list", "--left-right", "--count", "@{upstream}...HEAD"])).split() != ["0", "0"]:
        raise Generation5SourceGoldError("Repository must be upstream-even.")
    commit = str(_git(["rev-parse", "HEAD"]))
    body = _git(["show", f"{commit}:{MODULE_NAME}"], binary=True)
    if not isinstance(body, bytes) or hashlib.sha256(body).hexdigest() != sha256_file(Path(__file__).resolve()):
        raise Generation5SourceGoldError("Committed gold module drifted.")
    return {"commit": commit, "module_sha256": hashlib.sha256(body).hexdigest(),
            "clean": True, "upstream_ahead": 0, "upstream_behind": 0}


def _s1_preview() -> dict[str, Any]:
    replay = s1.replay_generation5_source_review(S1_PREVIEW_SHA256)
    paths = s1._paths(s1.DEFAULT_RUNTIME_ROOT, S1_PREVIEW_SHA256)
    if replay.get("idempotent_replay") is not True or sha256_file(paths["manifest"]) != S1_MANIFEST_SHA256:
        raise Generation5SourceGoldError("S1 review authority drifted.")
    preview = _read_json(paths["manifest"]).get("preview")
    if not isinstance(preview, dict) or preview.get("speaker_label_count") != 40:
        raise Generation5SourceGoldError("S1 review denominator drifted.")
    return preview


def parse_review_answers(answer_text: str, expected_refs: Sequence[str]) -> dict[str, str]:
    expected = list(expected_refs)
    answers: dict[str, str] = {}
    for raw in str(answer_text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if "=" not in line:
            raise Generation5SourceGoldError("Every answer line must contain '='.")
        reference, value = (part.strip() for part in line.split("=", 1))
        if reference not in expected or reference in answers:
            raise Generation5SourceGoldError("Answer references are unknown or duplicated.")
        if not value or _normalized_identity(value) in {"unknown", "unanswered", "not sure", "unsure"}:
            raise Generation5SourceGoldError("Every speaker needs a stable identity or alias.")
        answers[reference] = _canonical_identity(value)
    if set(answers) != set(expected):
        raise Generation5SourceGoldError("The private speaker review is incomplete.")
    return answers


def _person_id(identity: str) -> str:
    return "g5s-person-" + hashlib.sha256(_normalized_identity(identity).encode()).hexdigest()[:24]


def _cases(
    authority: Mapping[str, Any], answers: Mapping[str, str], enrolled: Mapping[str, str],
    *, context_derived_refs: set[str],
) -> list[dict[str, Any]]:
    private = authority.get("private_evidence")
    cards = private.get("cards") if isinstance(private, Mapping) else None
    membership = private.get("membership") if isinstance(private, Mapping) else None
    if not isinstance(cards, list) or not isinstance(membership, list):
        raise Generation5SourceGoldError("S1 private membership is unavailable.")
    member_by_ordinal = {int(row["ordinal"]): row for row in membership if isinstance(row, Mapping)}
    cases: dict[int, dict[str, Any]] = {}
    for card in cards:
        if not isinstance(card, Mapping):
            raise Generation5SourceGoldError("A private review card is invalid.")
        ordinal = int(card.get("ordinal") or 0)
        member = member_by_ordinal.get(ordinal)
        reference = str(card.get("speaker_ref") or "")
        label = str(card.get("speaker_label") or "")
        if not member or reference not in answers or not label:
            raise Generation5SourceGoldError("Private review membership drifted.")
        source = str(member.get("source_sha256") or "")
        transcript = str(member.get("provider_result_sha256") or "")
        case_id = "g5s-case-" + _canonical_hash({"source": source, "transcript": transcript})[:20]
        case = cases.setdefault(ordinal, {
            "case_id": case_id, "enumerated_ordinal": ordinal,
            "source_sha256": source, "transcript_sha256": transcript,
            "conversation_id": "g5s-conversation-" + hashlib.sha256(case_id.encode()).hexdigest()[:24],
            "recording_id": "g5s-recording-" + hashlib.sha256(source.encode()).hexdigest()[:24],
            "speaker_gold": [], "overlap_codes": [],
        })
        identity = answers[reference]
        normalized = _normalized_identity(identity)
        case["speaker_gold"].append({
            "speaker_label": label, "person_id": _person_id(identity),
            "enrolled_subject_id": enrolled.get(normalized, ""),
            "private_identity_display": identity,
            "identity_authority": "transcript_context_derived" if reference in context_derived_refs else "operator_supplied",
            "answer_sha256": hashlib.sha256(f"{reference}={identity}".encode()).hexdigest(),
        })
    result = [cases[key] for key in sorted(cases)]
    if len(result) != 12 or any(not CASE_RE.fullmatch(case["case_id"]) for case in result):
        raise Generation5SourceGoldError("The case denominator drifted.")
    return result


def evaluate_population(cases: Sequence[Mapping[str, Any]], expected_sources: set[str]) -> dict[str, Any]:
    conversations, recordings, sources, transcripts = set(), set(), set(), set()
    people_sessions: dict[str, set[str]] = defaultdict(set)
    enrolled_sessions: dict[str, set[str]] = defaultdict(set)
    overlap_codes: set[str] = set()
    complete = len(cases) == 7
    for case in cases:
        source, transcript = str(case.get("source_sha256") or ""), str(case.get("transcript_sha256") or "")
        conversation, recording = str(case.get("conversation_id") or ""), str(case.get("recording_id") or "")
        speakers = case.get("speaker_gold")
        if not source or not transcript or not conversation or not recording or not isinstance(speakers, list) or not speakers:
            complete = False
            continue
        for value, seen, code in ((source, sources, "duplicate_source"), (transcript, transcripts, "duplicate_derivative"),
                                  (conversation, conversations, "duplicate_conversation"), (recording, recordings, "duplicate_recording")):
            if value in seen:
                overlap_codes.add(code)
            seen.add(value)
        for speaker in speakers:
            if not isinstance(speaker, Mapping) or not speaker.get("person_id"):
                complete = False
                continue
            people_sessions[str(speaker["person_id"])].add(conversation)
            subject = str(speaker.get("enrolled_subject_id") or "")
            if subject:
                enrolled_sessions[subject].add(conversation)
        overlap_codes.update(str(code) for code in case.get("overlap_codes") or [])
    if sources - expected_sources:
        overlap_codes.add("source_outside_authority")
    pair_count = sum(len(tuple(combinations(sessions, 2))) for sessions in people_sessions.values())
    gates = {
        "exactly_seven_conversations": len(conversations) == 7,
        "both_enrolled_people_have_two_recordings": sum(len(v) >= 2 for v in enrolled_sessions.values()) == 2,
        "minimum_five_people": len(people_sessions) >= 5,
        "minimum_four_same_person_session_pairs": pair_count >= 4,
        "complete_private_gold": complete,
        "zero_overlap": not overlap_codes,
        "all_sources_within_authority": sources <= expected_sources,
    }
    return {"conversation_count": len(conversations), "recording_count": len(recordings),
            "person_count": len(people_sessions), "enrolled_people_with_two_recordings_count": sum(len(v) >= 2 for v in enrolled_sessions.values()),
            "same_person_session_pair_count": pair_count, "complete_gold_case_count": len(cases) if complete else 0,
            "overlap_count": len(overlap_codes), "overlap_code_set_sha256": _canonical_hash(sorted(overlap_codes)),
            "gates": gates, "passing": all(gates.values())}


def select_first_passing(cases: Sequence[Mapping[str, Any]], expected_sources: set[str]) -> tuple[list[dict[str, Any]], dict[str, Any] | None, int]:
    by_ordinal = {int(case["enumerated_ordinal"]): dict(case) for case in cases}
    checked = 0
    for extra_ordinals in combinations(range(3, 13), 5):
        checked += 1
        subset = [by_ordinal[ordinal] for ordinal in (1, 2, *extra_ordinals)]
        result = evaluate_population(subset, expected_sources)
        if result["passing"]:
            return subset, result, checked
    return [], None, checked


def preview_generation5_source_gold(
    answer_text: str, *, context_derived_answers: Mapping[str, str] | None = None,
    s1_preview: Mapping[str, Any] | None = None,
    enrolled_identity_map: Mapping[str, str] | None = None,
    repository_authority: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    authority = dict(s1_preview or _s1_preview())
    cards = authority.get("private_evidence", {}).get("cards")
    if not isinstance(cards, list) or len(cards) != 40:
        raise Generation5SourceGoldError("S1 review cards are incomplete.")
    refs = [str(card.get("speaker_ref") or "") for card in cards]
    context = {str(key): _canonical_identity(value) for key, value in (context_derived_answers or {}).items()}
    if set(context) - set(refs):
        raise Generation5SourceGoldError("A context-derived answer reference is unknown.")
    supplied_lines = [line for line in answer_text.splitlines() if line.strip()]
    supplied_refs = {line.split("=", 1)[0].strip() for line in supplied_lines if "=" in line}
    completed_text = answer_text.rstrip() + "\n" + "\n".join(
        f"{reference} = {identity}" for reference, identity in context.items() if reference not in supplied_refs
    )
    answers = parse_review_answers(completed_text, refs)
    enrolled = dict(enrolled_identity_map or prior_gold._enrolled_identity_map())
    cases = _cases(authority, answers, enrolled, context_derived_refs=set(context))
    expected_sources = {case["source_sha256"] for case in cases}
    selected, population, checked = select_first_passing(cases, expected_sources)
    passing = bool(selected and population and population["passing"])
    selected_ids = [case["case_id"] for case in selected]
    answer_projection = sorted(
        ({"speaker_ref": ref, "identity_sha256": hashlib.sha256(_normalized_identity(value).encode()).hexdigest(),
          "identity_authority": "transcript_context_derived" if ref in context else "operator_supplied"}
         for ref, value in answers.items()), key=lambda item: item["speaker_ref"],
    )
    core = {
        "schema_version": PREVIEW_SCHEMA,
        "status": "ready_for_independent_j1_review" if passing else "population_infeasible_stop",
        "repository_authority": dict(repository_authority or _repository_authority()),
        "s1_preview_sha256": S1_PREVIEW_SHA256, "s1_manifest_sha256": S1_MANIFEST_SHA256,
        "reviewed_candidate_count": 12, "reviewed_speaker_label_count": 40,
        "operator_supplied_answer_count": 40 - len(context), "context_derived_answer_count": len(context),
        "answer_set_sha256": _canonical_hash(answer_projection), "combination_size": 7,
        "required_ordinals": [1, 2], "combinations_checked": checked,
        "population_feasible": passing, "population_result": population or {},
        "selected_case_ids_sha256": _canonical_hash(selected_ids),
        "selected_source_set_sha256": _canonical_hash(sorted(case["source_sha256"] for case in selected)),
        "selected_transcript_set_sha256": _canonical_hash(sorted(case["transcript_sha256"] for case in selected)),
        "private_gold": {"all_cases": cases, "selected_cases": selected, "selected_case_ids": selected_ids},
        "action_vector": {"submit_exact_population_and_gold_to_j1": passing, "freeze_cohort_or_gold": False,
                          "run_models_or_predictions": False, "reveal_gold_to_workers": False,
                          "mutate_profiles_or_references": False, "enable_default_integration": False,
                          "run_historical_reprocessing": False},
        "contains_private_membership": True, "contains_identity_names_or_aliases": True,
        "contains_transcript_text": False, "contains_audio": False,
        "did_freeze_cohort_or_gold": False, "did_run_models_or_predictions": False,
        "did_reveal_gold_to_workers": False,
    }
    return {**core, "content_sha256": _canonical_hash(core)}


def _portable(preview: Mapping[str, Any]) -> dict[str, Any]:
    return {key: preview[key] for key in (
        "schema_version", "status", "answer_set_sha256", "reviewed_candidate_count",
        "reviewed_speaker_label_count", "operator_supplied_answer_count", "context_derived_answer_count",
        "combination_size", "required_ordinals", "combinations_checked", "population_feasible",
        "population_result", "selected_case_ids_sha256", "selected_source_set_sha256",
        "selected_transcript_set_sha256", "action_vector", "did_freeze_cohort_or_gold",
        "did_run_models_or_predictions", "did_reveal_gold_to_workers",
    )}


def _paths(runtime_root: Path, content_sha256: str) -> dict[str, Path]:
    root = runtime_root.expanduser().absolute()
    run = root / f"generation5-source-gold-{content_sha256[:24]}"
    return {"root": root, "run": run, "manifest": run / "private-manifest.json", "receipt": run / "receipt.json"}


def apply_generation5_source_gold(reviewed_preview: Mapping[str, Any], *, expected_content_sha256: str,
                                  runtime_root: Path = DEFAULT_RUNTIME_ROOT) -> dict[str, Any]:
    preview = dict(reviewed_preview)
    core = {key: value for key, value in preview.items() if key != "content_sha256"}
    if preview.get("content_sha256") != expected_content_sha256 or _canonical_hash(core) != expected_content_sha256:
        raise Generation5SourceGoldError("Reviewed gold proposal is stale.")
    if preview.get("repository_authority") != _repository_authority():
        raise Generation5SourceGoldError("Reviewed repository authority is stale.")
    paths = _paths(runtime_root, expected_content_sha256)
    if paths["receipt"].exists():
        return replay_generation5_source_gold(expected_content_sha256, runtime_root=runtime_root)
    ensure_private_tree(paths["root"], paths["run"])
    write_immutable_private_json(paths["manifest"], {"schema_version": MANIFEST_SCHEMA, "status": "frozen_proposal", "preview": preview})
    receipt = {**_portable(preview), "schema_version": RECEIPT_SCHEMA,
               "preview_content_sha256": expected_content_sha256, "manifest_sha256": sha256_file(paths["manifest"]), "mode": "0600"}
    write_immutable_private_json(paths["receipt"], receipt)
    return {**receipt, "idempotent_replay": False}


def replay_generation5_source_gold(expected_content_sha256: str, *, runtime_root: Path = DEFAULT_RUNTIME_ROOT) -> dict[str, Any]:
    paths = _paths(runtime_root, expected_content_sha256)
    require_private_file(paths["manifest"], paths["root"])
    require_private_file(paths["receipt"], paths["root"])
    manifest, receipt = _read_json(paths["manifest"]), _read_json(paths["receipt"])
    preview = manifest.get("preview")
    if not isinstance(preview, Mapping):
        raise Generation5SourceGoldError("Private gold proposal is missing.")
    core = {key: value for key, value in preview.items() if key != "content_sha256"}
    expected_receipt = {**_portable(preview), "schema_version": RECEIPT_SCHEMA,
                        "preview_content_sha256": expected_content_sha256,
                        "manifest_sha256": sha256_file(paths["manifest"]), "mode": "0600"}
    if (_canonical_hash(core) != expected_content_sha256 or preview.get("content_sha256") != expected_content_sha256
            or manifest != {"schema_version": MANIFEST_SCHEMA, "status": "frozen_proposal", "preview": preview}
            or receipt != expected_receipt):
        raise Generation5SourceGoldError("Private gold proposal drifted.")
    return {**receipt, "replay_schema_version": REPLAY_SCHEMA, "idempotent_replay": True}
