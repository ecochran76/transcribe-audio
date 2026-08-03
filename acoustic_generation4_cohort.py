"""Plan 0052 G1A cohort and private-gold feasibility authority.

The portable surface contains counts and hashes only.  Exact source,
transcript, speaker-label, and gold membership remains in the private G1A
manifest.  This module can request the plan's one supplemental pool, but it
cannot freeze either cohort membership or gold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import acoustic_generation4_campaign as campaign
import acoustic_generation4_media as media
from acoustic_audio_derivatives import (
    ensure_private_tree,
    require_private_file,
    sha256_file,
    write_immutable_private_json,
)


PREVIEW_SCHEMA = "transcribe-audio.generation4-cohort-g1a-preview.v1"
MANIFEST_SCHEMA = "transcribe-audio.generation4-cohort-g1a-manifest.v1"
RECEIPT_SCHEMA = "transcribe-audio.generation4-cohort-g1a-receipt.v1"
REPLAY_SCHEMA = "transcribe-audio.generation4-cohort-g1a-replay.v1"
GOLD_SCHEMA = "transcribe-audio.generation4-private-gold-feasibility.v1"
G0_PREVIEW_SHA256 = (
    "aa179741e735247e87cc6143c6526669670734c8c562ed166160eb0c6d605010"
)
G0_MANIFEST_SHA256 = (
    "ad9e26b59502508c8810e11648d519d99860579aea1ca731445459b196836d22"
)
MEDIA_PREVIEW_SHA256 = campaign.MEDIA_PREVIEW_SHA256
QUALIFIED_SET_SHA256 = (
    "e3c908f80c922365ead50795728feb959d8aa93e542ee2882be79efc456e48be"
)
ORIGINAL_MEDIA_MANIFEST_SHA256 = (
    "8b115bb92930916b087f114ab396f43f08d40b39f5faff8e1254d30a709c29fe"
)
SUPPLEMENTAL_MEDIA_PREVIEW_SHA256 = (
    "cc405f40414f69bea012559d5ca4c10098ed4ab0d4e4efc37264a361c26f82d9"
)
SUPPLEMENTAL_MEDIA_MANIFEST_SHA256 = (
    "c34a4ebd2d78fef8193aec18f15c97146f06f99c32d2c29d81066719954ab677"
)
SUPPLEMENTAL_QUALIFIED_SET_SHA256 = (
    "09ae99141880df95b3531563b484008ea411ccafab411b4da311627d5e16d994"
)
COMBINED_QUALIFIED_SET_SHA256 = (
    "460fa3dd3befa17e249860b70477474580202577a599a357e7b7c641609cd4c2"
)
SUPPLEMENTAL_MEDIA_RUNTIME_ROOT = Path(
    "~/.local/state/transcribe-audio/plan-0052/g1a/supplemental-media"
)
DEFAULT_RUNTIME_ROOT = Path("~/.local/state/transcribe-audio/plan-0052/g1a")
MAX_SUPPLEMENTAL_CANDIDATES = 12
MODULE_NAME = "acoustic_generation4_cohort.py"
COMMIT_RE = re.compile(r"[a-f0-9]{40}")
SHA256_RE = re.compile(r"[a-f0-9]{64}")


class Generation4CohortError(ValueError):
    """Raised when G1A evidence cannot fail closed."""


def _canonical_hash(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Generation4CohortError("Private G1A input is unreadable.") from exc
    if not isinstance(value, dict):
        raise Generation4CohortError("Private G1A input must be an object.")
    return value


def _git(args: Sequence[str], *, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", *args],
        cwd=Path(__file__).resolve().parent,
        check=False,
        capture_output=True,
        text=not binary,
    )
    if result.returncode:
        raise Generation4CohortError("G1A repository authority is unavailable.")
    return result.stdout if binary else result.stdout.strip()


def _validated_repository(value: Mapping[str, Any]) -> dict[str, Any]:
    repository = dict(value)
    if (
        not COMMIT_RE.fullmatch(str(repository.get("commit") or ""))
        or repository.get("module_name") != MODULE_NAME
        or not SHA256_RE.fullmatch(str(repository.get("module_sha256") or ""))
        or repository.get("clean") is not True
        or repository.get("upstream_ahead") != 0
        or repository.get("upstream_behind") != 0
    ):
        raise Generation4CohortError("G1A repository authority is invalid.")
    return repository


def _repository_authority() -> dict[str, Any]:
    if _git(["status", "--porcelain=v1", "--untracked-files=normal"]):
        raise Generation4CohortError("Repository must be clean for G1A authority.")
    behind_ahead = str(
        _git(["rev-list", "--left-right", "--count", "@{upstream}...HEAD"])
    ).split()
    if behind_ahead != ["0", "0"]:
        raise Generation4CohortError("Repository must be upstream-even for G1A authority.")
    commit = str(_git(["log", "-1", "--format=%H", "--", MODULE_NAME]))
    if not COMMIT_RE.fullmatch(commit) or _git(
        ["merge-base", "--is-ancestor", commit, "HEAD"]
    ) != "":
        raise Generation4CohortError("G1A module commit is not an ancestor of HEAD.")
    module_blob = _git(["show", f"{commit}:{MODULE_NAME}"], binary=True)
    if not isinstance(module_blob, bytes):
        raise Generation4CohortError("G1A repository blob is unavailable.")
    module_sha256 = hashlib.sha256(module_blob).hexdigest()
    if sha256_file(Path(__file__).resolve()) != module_sha256:
        raise Generation4CohortError("G1A module authority drifted.")
    return _validated_repository(
        {
            "commit": commit,
            "module_name": MODULE_NAME,
            "module_sha256": module_sha256,
            "clean": True,
            "upstream_ahead": 0,
            "upstream_behind": 0,
        }
    )


def _g0_authority() -> dict[str, Any]:
    paths = campaign._paths(campaign.DEFAULT_RUNTIME_ROOT, G0_PREVIEW_SHA256)
    require_private_file(paths["manifest"], paths["root"].expanduser().absolute())
    manifest = _read_json(paths["manifest"])
    preview = manifest.get("preview")
    if (
        sha256_file(paths["manifest"]) != G0_MANIFEST_SHA256
        or not isinstance(preview, Mapping)
        or preview.get("content_sha256") != G0_PREVIEW_SHA256
        or preview.get("action_vector", {}).get("run_g1a_cohort_gold_feasibility")
        is not True
        or preview.get("action_vector", {}).get("freeze_g2_envelope") is not False
    ):
        raise Generation4CohortError("Frozen G0 authority did not authorize only G1A work.")
    inherited = preview.get("inherited_evidence", {}).get("media", {})
    if (
        inherited.get("preview_content_sha256") != MEDIA_PREVIEW_SHA256
        or inherited.get("qualified_set_sha256") != QUALIFIED_SET_SHA256
        or inherited.get("qualified_count") != 10
        or inherited.get("idempotent_replay") is not True
    ):
        raise Generation4CohortError("G0 does not bind the exact Plan 0051 pool.")
    return {
        "g0_preview_sha256": G0_PREVIEW_SHA256,
        "g0_manifest_sha256": G0_MANIFEST_SHA256,
        "media_preview_sha256": MEDIA_PREVIEW_SHA256,
        "qualified_set_sha256": QUALIFIED_SET_SHA256,
    }


def _media_membership() -> list[dict[str, Any]]:
    paths = media._paths(media.DEFAULT_RUNTIME_ROOT, MEDIA_PREVIEW_SHA256)
    require_private_file(paths["manifest"], paths["root"].expanduser().absolute())
    manifest = _read_json(paths["manifest"])
    preview = manifest.get("preview")
    if not isinstance(preview, Mapping):
        raise Generation4CohortError("Plan 0051 private membership is unavailable.")
    qualified = [
        dict(item)
        for item in preview.get("private_results", [])
        if isinstance(item, Mapping) and item.get("status") == "qualified"
    ]
    qualified_hash = _canonical_hash(
        sorted(item.get("source_sha256") for item in qualified)
    )
    if len(qualified) != 10 or qualified_hash != QUALIFIED_SET_SHA256:
        raise Generation4CohortError("Plan 0051 qualified membership drifted.")
    return qualified


def _manifest_membership(
    manifest_path: Path,
    *,
    private_root: Path,
    expected_manifest_sha256: str,
    expected_preview_sha256: str,
    expected_qualified_set_sha256: str,
    expected_qualified_count: int,
    expected_candidate_count: int,
    authority_origin: str,
) -> list[dict[str, Any]]:
    """Load an exact qualified set from one immutable private media manifest."""
    require_private_file(manifest_path, private_root.expanduser().absolute())
    manifest = _read_json(manifest_path)
    preview = manifest.get("preview")
    if (
        sha256_file(manifest_path) != expected_manifest_sha256
        or manifest.get("schema_version") != media.MANIFEST_SCHEMA
        or manifest.get("status") != "frozen"
        or not isinstance(preview, Mapping)
        or preview.get("content_sha256") != expected_preview_sha256
        or preview.get("candidate_count") != expected_candidate_count
        or preview.get("qualified_count") != expected_qualified_count
        or expected_candidate_count > MAX_SUPPLEMENTAL_CANDIDATES
    ):
        raise Generation4CohortError("Frozen source media manifest drifted.")
    private_results = preview.get("private_results")
    if (
        not isinstance(private_results, list)
        or len(private_results) != expected_candidate_count
    ):
        raise Generation4CohortError("Frozen source membership is incomplete.")
    qualified: list[dict[str, Any]] = []
    hashes: set[str] = set()
    for raw in private_results:
        if not isinstance(raw, Mapping):
            raise Generation4CohortError("Frozen source membership is invalid.")
        item = dict(raw)
        if item.get("status") != "qualified":
            continue
        source_sha256 = str(item.get("source_sha256") or "")
        path = str(item.get("path") or "")
        if (
            not SHA256_RE.fullmatch(source_sha256)
            or not path
            or source_sha256 in hashes
        ):
            raise Generation4CohortError("Frozen qualified membership is invalid.")
        hashes.add(source_sha256)
        item["authority_origin"] = authority_origin
        qualified.append(item)
    if (
        len(qualified) != expected_qualified_count
        or _canonical_hash(sorted(hashes)) != expected_qualified_set_sha256
    ):
        raise Generation4CohortError("Frozen qualified set hash drifted.")
    return qualified


def _source_authority() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Resolve the exact original plus one consumed supplemental source union."""
    g0 = _g0_authority()
    original_paths = media._paths(media.DEFAULT_RUNTIME_ROOT, MEDIA_PREVIEW_SHA256)
    original = _manifest_membership(
        original_paths["manifest"],
        private_root=original_paths["root"],
        expected_manifest_sha256=ORIGINAL_MEDIA_MANIFEST_SHA256,
        expected_preview_sha256=MEDIA_PREVIEW_SHA256,
        expected_qualified_set_sha256=QUALIFIED_SET_SHA256,
        expected_qualified_count=10,
        expected_candidate_count=12,
        authority_origin="original",
    )
    supplemental_paths = media._paths(
        SUPPLEMENTAL_MEDIA_RUNTIME_ROOT, SUPPLEMENTAL_MEDIA_PREVIEW_SHA256
    )
    supplemental = _manifest_membership(
        supplemental_paths["manifest"],
        private_root=supplemental_paths["root"],
        expected_manifest_sha256=SUPPLEMENTAL_MEDIA_MANIFEST_SHA256,
        expected_preview_sha256=SUPPLEMENTAL_MEDIA_PREVIEW_SHA256,
        expected_qualified_set_sha256=SUPPLEMENTAL_QUALIFIED_SET_SHA256,
        expected_qualified_count=12,
        expected_candidate_count=12,
        authority_origin="supplemental",
    )
    original_hashes = {str(item["source_sha256"]) for item in original}
    supplemental_hashes = {str(item["source_sha256"]) for item in supplemental}
    if original_hashes & supplemental_hashes:
        raise Generation4CohortError("Original and supplemental sources overlap.")
    combined = original + supplemental
    combined_hash = _canonical_hash(
        sorted(str(item["source_sha256"]) for item in combined)
    )
    if combined_hash != COMBINED_QUALIFIED_SET_SHA256:
        raise Generation4CohortError("Combined qualified source authority drifted.")
    authority = {
        **g0,
        "original_media_manifest_sha256": ORIGINAL_MEDIA_MANIFEST_SHA256,
        "original_qualified_set_sha256": QUALIFIED_SET_SHA256,
        "original_qualified_count": len(original),
        "supplemental_media_preview_sha256": SUPPLEMENTAL_MEDIA_PREVIEW_SHA256,
        "supplemental_media_manifest_sha256": SUPPLEMENTAL_MEDIA_MANIFEST_SHA256,
        "supplemental_qualified_set_sha256": SUPPLEMENTAL_QUALIFIED_SET_SHA256,
        "supplemental_qualified_count": len(supplemental),
        "supplemental_pool_count": 1,
        "supplemental_pool_consumed": True,
        "qualified_set_sha256": combined_hash,
        "qualified_count": len(combined),
        "authority_origins": ["original", "supplemental"],
    }
    return authority, combined


def _transcript_rows(source_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    resolved_root = source_root.expanduser().resolve(strict=True)
    for path in sorted(source_root.glob("*.json")):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(value, dict) or not value.get("source_media_path"):
            continue
        utterances = value.get("utterances")
        if not isinstance(utterances, list):
            continue
        labels = sorted(
            {
                str(item.get("speaker") or "").strip()
                for item in utterances
                if isinstance(item, Mapping) and str(item.get("speaker") or "").strip()
            }
        )
        source_candidates = [
            Path(str(value.get(key) or "")).expanduser()
            for key in ("source_media_path", "working_media_path")
            if str(value.get(key) or "").strip()
        ]
        resolved_sources: list[Path] = []
        for source in source_candidates:
            try:
                candidate = source.resolve(strict=True)
            except OSError:
                continue
            try:
                relative = candidate.relative_to(resolved_root)
            except ValueError:
                continue
            if (
                not source.is_symlink()
                and candidate.is_file()
                and len(relative.parts) == 1
                and candidate not in resolved_sources
            ):
                resolved_sources.append(candidate)
        if not resolved_sources:
            continue
        event = value.get("event") if isinstance(value.get("event"), Mapping) else {}
        transcript_sha256 = sha256_file(path)
        for resolved_source in resolved_sources:
            rows.append(
                {
                    "source_path": str(resolved_source),
                    "source_sha256": sha256_file(resolved_source),
                    "transcript_path": str(path.resolve()),
                    "transcript_sha256": transcript_sha256,
                    "speaker_labels": labels,
                    "speaker_label_count": len(labels),
                    "conversation_id": str(value.get("conversation_id") or ""),
                    "recording_id": str(value.get("recording_id") or ""),
                    "has_calendar_context": bool(
                        event.get("attendees") or event.get("participants")
                    ),
                }
            )
    return rows


def _gold_cases(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    if path.is_symlink() or path.stat().st_mode & 0o077:
        raise Generation4CohortError("Private gold feasibility input must be a 0600 file.")
    value = _read_json(path)
    cases = value.get("cases")
    if value.get("schema_version") != GOLD_SCHEMA or not isinstance(cases, list):
        raise Generation4CohortError("Private gold feasibility input has the wrong schema.")
    return [dict(item) for item in cases if isinstance(item, Mapping)]


def evaluate_population(
    cases: Sequence[Mapping[str, Any]],
    *,
    expected_sources: set[str],
) -> dict[str, Any]:
    """Evaluate population gates from opaque private gold records."""
    conversations: set[str] = set()
    recordings: set[str] = set()
    people_sessions: dict[str, set[str]] = defaultdict(set)
    enrolled_sessions: dict[str, set[str]] = defaultdict(set)
    complete_labels = True
    source_hashes: set[str] = set()
    transcript_hashes: set[str] = set()
    overlap_codes: set[str] = set()
    for case in cases:
        source_hash = str(case.get("source_sha256") or "")
        transcript_hash = str(case.get("transcript_sha256") or "")
        conversation = str(case.get("conversation_id") or "")
        recording = str(case.get("recording_id") or "")
        speakers = case.get("speaker_gold")
        if (
            not source_hash
            or not transcript_hash
            or not conversation
            or not recording
            or not isinstance(speakers, list)
            or not speakers
        ):
            complete_labels = False
            continue
        if source_hash in source_hashes:
            overlap_codes.add("duplicate_source")
        if transcript_hash in transcript_hashes:
            overlap_codes.add("duplicate_derivative")
        if conversation in conversations:
            overlap_codes.add("duplicate_conversation")
        if recording in recordings:
            overlap_codes.add("duplicate_recording")
        source_hashes.add(source_hash)
        transcript_hashes.add(transcript_hash)
        conversations.add(conversation)
        recordings.add(recording)
        for speaker in speakers:
            if not isinstance(speaker, Mapping):
                complete_labels = False
                continue
            label = str(speaker.get("speaker_label") or "")
            person = str(speaker.get("person_id") or "")
            if not label or not person:
                complete_labels = False
                continue
            people_sessions[person].add(conversation)
            enrolled = str(speaker.get("enrolled_subject_id") or "")
            if enrolled:
                enrolled_sessions[enrolled].add(conversation)
        for code in case.get("overlap_codes") or []:
            overlap_codes.add(str(code))
    if source_hashes - expected_sources:
        overlap_codes.add("source_outside_authority")
    session_pair_count = sum(
        len(tuple(combinations(sessions, 2))) for sessions in people_sessions.values()
    )
    enrolled_with_two = sum(len(sessions) >= 2 for sessions in enrolled_sessions.values())
    gates = {
        "minimum_seven_conversations": len(conversations) >= 7,
        "both_enrolled_people_have_two_sessions": enrolled_with_two >= 2,
        "minimum_five_people": len(people_sessions) >= 5,
        "minimum_four_same_person_session_pairs": session_pair_count >= 4,
        "complete_private_gold": complete_labels and bool(cases),
        "zero_overlap": not overlap_codes,
        "all_sources_within_authority": source_hashes <= expected_sources,
    }
    return {
        "conversation_count": len(conversations),
        "recording_count": len(recordings),
        "person_count": len(people_sessions),
        "enrolled_people_with_two_sessions_count": enrolled_with_two,
        "same_person_session_pair_count": session_pair_count,
        "complete_gold_case_count": len(cases) if complete_labels else 0,
        "overlap_count": len(overlap_codes),
        "overlap_code_set_sha256": _canonical_hash(sorted(overlap_codes)),
        "gates": gates,
        "passing": all(gates.values()),
    }


def _supplemental_candidates(
    rows: Sequence[Mapping[str, Any]], excluded_source_hashes: set[str]
) -> list[dict[str, Any]]:
    eligible = []
    seen_transcripts: set[str] = set()
    for row in rows:
        transcript_hash = str(row.get("transcript_sha256") or "")
        if (
            row.get("speaker_label_count", 0) <= 0
            or row.get("source_sha256") in excluded_source_hashes
            or transcript_hash in seen_transcripts
        ):
            continue
        eligible.append(dict(row))
        seen_transcripts.add(transcript_hash)
    eligible.sort(
        key=lambda row: (
            not bool(row.get("has_calendar_context")),
            -int(row.get("speaker_label_count") or 0),
            str(row.get("source_sha256") or ""),
        )
    )
    return eligible[:MAX_SUPPLEMENTAL_CANDIDATES]


def _passing_subset(
    cases: Sequence[Mapping[str, Any]], *, expected_sources: set[str]
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Return the smallest deterministic 7+ gold subset passing every gate."""
    ordered = sorted(
        (dict(case) for case in cases),
        key=lambda case: (
            str(case.get("source_sha256") or ""),
            str(case.get("transcript_sha256") or ""),
        ),
    )
    for size in range(7, len(ordered) + 1):
        for subset in combinations(ordered, size):
            result = evaluate_population(subset, expected_sources=expected_sources)
            if result["passing"]:
                return list(subset), result
    return [], None


def _portable(preview: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": RECEIPT_SCHEMA,
        "status": preview["status"],
        "g0_preview_sha256": preview["authority"]["g0_preview_sha256"],
        "g0_manifest_sha256": preview["authority"]["g0_manifest_sha256"],
        "media_preview_sha256": preview["authority"]["media_preview_sha256"],
        "qualified_set_sha256": preview["authority"]["qualified_set_sha256"],
        "source_authority": dict(preview["authority"]),
        "repository_authority": dict(preview["repository_authority"]),
        "delegation_receipt": dict(preview["delegation_receipt"]),
        "qualified_recording_count": preview["qualified_recording_count"],
        "qualified_with_exact_transcript_count": preview[
            "qualified_with_exact_transcript_count"
        ],
        "qualified_without_exact_transcript_count": preview[
            "qualified_without_exact_transcript_count"
        ],
        "proposed_original_cohort_count": preview[
            "proposed_original_cohort_count"
        ],
        "population_feasibility": dict(preview["population_feasibility"]),
        "supplemental_candidate_count": preview["supplemental_candidate_count"],
        "supplemental_candidate_set_sha256": preview[
            "supplemental_candidate_set_sha256"
        ],
        "population": dict(preview["population"]),
        "action_vector": dict(preview["action_vector"]),
        "contains_paths": False,
        "contains_private_membership": False,
        "contains_speaker_labels": False,
        "contains_person_identifiers": False,
        "contains_transcript_text": False,
        "did_read_private_gold": preview["did_read_private_gold"],
        "did_load_or_run_models": False,
        "did_freeze_cohort_or_gold": False,
        "will_perform_external_write": False,
    }


def preview_generation4_cohort(
    *,
    source_root: Path = media.DEFAULT_SOURCE_ROOT,
    gold_path: Path | None = None,
    repository_authority: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    authority_value, qualified = _source_authority()
    repository_value = _validated_repository(
        repository_authority if repository_authority is not None else _repository_authority()
    )
    rows = _transcript_rows(source_root)
    by_source = {str(row["source_path"]): row for row in rows}
    exact = [
        {**row, "authority_origin": item["authority_origin"]}
        for item in qualified
        if (row := by_source.get(str(Path(item["path"]).resolve())))
    ]
    expected_sources = {str(item["source_sha256"]) for item in qualified}
    gold = _gold_cases(gold_path)
    transcript_authority = {
        (str(row["source_sha256"]), str(row["transcript_sha256"])): row
        for row in exact
    }
    aligned_gold: list[dict[str, Any]] = []
    for item in gold:
        case = dict(item)
        overlap_codes = list(case.get("overlap_codes") or [])
        row = transcript_authority.get(
            (
                str(case.get("source_sha256") or ""),
                str(case.get("transcript_sha256") or ""),
            )
        )
        if row is None:
            overlap_codes.append("gold_outside_transcript_authority")
        else:
            gold_labels = {
                str(speaker.get("speaker_label") or "")
                for speaker in case.get("speaker_gold") or []
                if isinstance(speaker, Mapping)
            }
            if gold_labels != set(row["speaker_labels"]):
                overlap_codes.append("gold_label_set_mismatch")
        case["overlap_codes"] = sorted(set(overlap_codes))
        aligned_gold.append(case)
    passing_cases, passing_population = _passing_subset(
        aligned_gold, expected_sources=expected_sources
    )
    reviewed_keys = {
        (str(case.get("source_sha256") or ""), str(case.get("transcript_sha256") or ""))
        for case in aligned_gold
        if not case.get("overlap_codes")
    }
    exact_keys = set(transcript_authority)
    all_linked_gold_reviewed = bool(exact_keys) and exact_keys <= reviewed_keys
    ranked_exact = sorted(
        exact,
        key=lambda row: (
            not bool(row.get("has_calendar_context")),
            -int(row.get("speaker_label_count") or 0),
            str(row.get("source_sha256") or ""),
        ),
    )
    potential_original = ranked_exact[:7] if len(ranked_exact) >= 7 else []
    maximum_people_slots_in_proposal = sum(
        int(row.get("speaker_label_count") or 0) for row in potential_original
    )
    original_population_potential = (
        len(potential_original) >= 7 and maximum_people_slots_in_proposal >= 5
    )
    provided_population = evaluate_population(
        aligned_gold, expected_sources=expected_sources
    )
    has_invalid_gold = bool(aligned_gold) and provided_population["overlap_count"] > 0
    if passing_population is not None:
        status = "passing_population_proposal"
        population = passing_population
        proposed_original = passing_cases
    elif has_invalid_gold:
        status = "stop"
        population = provided_population
        proposed_original = []
    elif not original_population_potential or all_linked_gold_reviewed:
        status = "supplemental_pool_requested"
        population = provided_population
        proposed_original = []
    else:
        status = "private_gold_review_required"
        population = provided_population
        proposed_original = potential_original
    supplemental = (
        _supplemental_candidates(rows, expected_sources)
        if status == "supplemental_pool_requested"
        else []
    )
    actions = {
        "request_one_supplemental_pool": status == "supplemental_pool_requested",
        "submit_population_proposal_to_j1": status == "passing_population_proposal",
        "complete_private_gold_review": status == "private_gold_review_required",
        "freeze_g2_cohort": False,
        "freeze_private_gold": False,
        "load_or_run_models": False,
        "reveal_gold_to_prediction_workers": False,
        "mutate_profiles_or_references": False,
    }
    supplemental_hash = _canonical_hash(
        [
            {
                "source_sha256": row["source_sha256"],
                "transcript_sha256": row["transcript_sha256"],
                "speaker_label_count": row["speaker_label_count"],
            }
            for row in supplemental
        ]
    )
    private = {
        "qualified_membership": qualified,
        "exact_transcript_rows": exact,
        "proposed_original_cohort": proposed_original,
        "supplemental_candidates": supplemental,
        "gold_cases": aligned_gold,
    }
    safe_evidence_sha256 = _canonical_hash(
        {
            "status": status,
            "g0_preview_sha256": authority_value["g0_preview_sha256"],
            "qualified_set_sha256": authority_value["qualified_set_sha256"],
            "repository_module_sha256": repository_value["module_sha256"],
            "qualified_recording_count": len(qualified),
            "qualified_with_exact_transcript_count": len(exact),
            "proposed_original_cohort_count": len(proposed_original),
            "population": population,
            "supplemental_candidate_count": len(supplemental),
            "supplemental_candidate_set_sha256": supplemental_hash,
            "action_vector": actions,
        }
    )
    delegation_receipt = {
        "status": "spawned",
        "lane": "G1A",
        "runtime_handle": "/root/g1a_cohort_feasibility",
        "terminal_status": (
            "gold_review_gate"
            if status == "private_gold_review_required"
            else status
        ),
        "returned_evidence_sha256": safe_evidence_sha256,
        "primary_reconciliation": (
            "pending_gold_review"
            if status == "private_gold_review_required"
            else "pending_primary_review"
        ),
    }
    core = {
        "schema_version": PREVIEW_SCHEMA,
        "status": status,
        "authority": authority_value,
        "repository_authority": repository_value,
        "delegation_receipt": delegation_receipt,
        "discovery_method": "file-searcher-resolved bounded top-level metadata scan",
        "qualified_recording_count": len(qualified),
        "qualified_with_exact_transcript_count": len(exact),
        "qualified_without_exact_transcript_count": len(qualified) - len(exact),
        "proposed_original_cohort_count": len(proposed_original),
        "population_feasibility": {
            "original_linked_recording_count": len(exact),
            "minimum_cohort_size": 7,
            "maximum_people_slots_in_proposed_subset": maximum_people_slots_in_proposal,
            "minimum_people_required": 5,
            "minimum_enrolled_session_appearances_required": 4,
            "minimum_same_person_session_pairs_required": 4,
            "maximum_same_person_pair_capacity": (
                len(potential_original) * (len(potential_original) - 1) // 2
            ),
            "original_subset_has_population_capacity": original_population_potential,
            "all_linked_gold_reviewed": all_linked_gold_reviewed,
            "missing_gold_is_not_population_infeasibility": not aligned_gold,
            "identity_session_coverage_status": (
                "proven_by_private_gold_subset"
                if passing_population is not None
                else "unknown_pending_private_gold_review"
                if original_population_potential and not all_linked_gold_reviewed
                else "proven_infeasible_from_complete_private_evidence"
            ),
        },
        "supplemental_candidate_count": len(supplemental),
        "supplemental_candidate_set_sha256": supplemental_hash,
        "population": population,
        "action_vector": actions,
        "private_evidence": private,
        "contains_paths": True,
        "contains_private_membership": True,
        "contains_speaker_labels": True,
        "contains_person_identifiers": bool(aligned_gold),
        "contains_transcript_text": False,
        "did_read_private_gold": bool(aligned_gold),
        "did_load_or_run_models": False,
        "did_freeze_cohort_or_gold": False,
        "will_perform_external_write": False,
    }
    return {**core, "content_sha256": _canonical_hash(core)}


def _paths(runtime_root: Path, content_sha256: str) -> dict[str, Path]:
    root = runtime_root.expanduser().absolute()
    run = root / f"generation4-cohort-g1a-{content_sha256[:24]}"
    return {
        "root": root,
        "run": run,
        "manifest": run / "private-manifest.json",
        "receipt": run / "receipt.json",
    }


def apply_generation4_cohort(
    reviewed_preview: Mapping[str, Any],
    *,
    expected_content_sha256: str,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    source_root: Path = media.DEFAULT_SOURCE_ROOT,
    gold_path: Path | None = None,
    repository_authority: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    preview = preview_generation4_cohort(
        source_root=source_root,
        gold_path=gold_path,
        repository_authority=repository_authority,
    )
    if dict(reviewed_preview) != preview or preview["content_sha256"] != expected_content_sha256:
        raise Generation4CohortError("Reviewed G1A preview is stale.")
    paths = _paths(runtime_root, expected_content_sha256)
    if paths["manifest"].exists() or paths["receipt"].exists():
        return replay_generation4_cohort(
            expected_content_sha256,
            runtime_root=runtime_root,
            source_root=source_root,
            gold_path=gold_path,
            repository_authority=repository_authority,
        )
    ensure_private_tree(paths["root"], paths["run"])
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "status": "frozen_g1a_feasibility_only",
        "preview": preview,
        "contains_paths": True,
        "contains_private_membership": True,
        "did_freeze_cohort_or_gold": False,
    }
    write_immutable_private_json(paths["manifest"], manifest)
    receipt = {
        **_portable(preview),
        "preview_content_sha256": preview["content_sha256"],
        "manifest_sha256": sha256_file(paths["manifest"]),
        "mode": "0600",
    }
    write_immutable_private_json(paths["receipt"], receipt)
    return {**receipt, "idempotent_replay": False}


def replay_generation4_cohort(
    expected_content_sha256: str,
    *,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    source_root: Path = media.DEFAULT_SOURCE_ROOT,
    gold_path: Path | None = None,
    repository_authority: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    preview = preview_generation4_cohort(
        source_root=source_root,
        gold_path=gold_path,
        repository_authority=repository_authority,
    )
    if preview["content_sha256"] != expected_content_sha256:
        raise Generation4CohortError("Frozen G1A preview drifted.")
    paths = _paths(runtime_root, expected_content_sha256)
    require_private_file(paths["manifest"], paths["root"])
    require_private_file(paths["receipt"], paths["root"])
    manifest = _read_json(paths["manifest"])
    expected_manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "status": "frozen_g1a_feasibility_only",
        "preview": preview,
        "contains_paths": True,
        "contains_private_membership": True,
        "did_freeze_cohort_or_gold": False,
    }
    receipt = _read_json(paths["receipt"])
    expected_receipt = {
        **_portable(preview),
        "preview_content_sha256": preview["content_sha256"],
        "manifest_sha256": sha256_file(paths["manifest"]),
        "mode": "0600",
    }
    if manifest != expected_manifest or receipt != expected_receipt:
        raise Generation4CohortError("G1A feasibility authority drifted.")
    return {
        **receipt,
        "replay_schema_version": REPLAY_SCHEMA,
        "replay_mode": "full_private_metadata_replay_without_gold_reveal_or_models",
        "idempotent_replay": True,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plan 0052 G1A cohort feasibility")
    parser.add_argument("mode", choices=("preview", "apply", "replay"))
    parser.add_argument("--expected-content-sha256", default="")
    parser.add_argument("--gold-path", type=Path)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    args = parser.parse_args(argv)
    if args.mode == "preview":
        result = preview_generation4_cohort(gold_path=args.gold_path)
        result = {**_portable(result), "preview_content_sha256": result["content_sha256"]}
    elif args.mode == "replay":
        result = replay_generation4_cohort(
            args.expected_content_sha256,
            runtime_root=args.runtime_root,
            gold_path=args.gold_path,
        )
    else:
        preview = preview_generation4_cohort(gold_path=args.gold_path)
        result = apply_generation4_cohort(
            preview,
            expected_content_sha256=args.expected_content_sha256,
            runtime_root=args.runtime_root,
            gold_path=args.gold_path,
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
