"""Enumerate Plan 0054 E1 candidates and build a private gold-review bundle."""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import acoustic_generation5_recovery_authority as r0
import acoustic_generation5_recovery_j2_acceptance as j2
from acoustic_audio_derivatives import (
    ensure_private_tree,
    require_private_file,
    sha256_file,
    write_immutable_private_json,
)


PREVIEW_SCHEMA = "transcribe-audio.generation5-e1-review-preview.v1"
MANIFEST_SCHEMA = "transcribe-audio.generation5-e1-review-manifest.v1"
RECEIPT_SCHEMA = "transcribe-audio.generation5-e1-review-receipt.v1"
REPLAY_SCHEMA = "transcribe-audio.generation5-e1-review-replay.v1"
DEFAULT_RUNTIME_ROOT = Path("~/.local/state/transcribe-audio/plan-0054/e1")
R0_PREVIEW_SHA256 = "de59d3da1edce0e0e5e0050582cac442e39096bcb5ca30c1f57aa230e928d307"
R0_MANIFEST_SHA256 = "3fc8f06de8a098d8312fd7bc6dbe3f327dafce90524dff71c04542716641137e"
R0_SELECTED_MEMBERSHIP_SHA256 = "172477eac32dbca0d2f3ffe6599f6b30167b0685ef692bb7ddc4c819bf689eb5"
J2_PREVIEW_SHA256 = "52321890681eb56a5ee515aae5abcf708984ac7fa80f0e5886953d7a480b7a54"
J2_MANIFEST_SHA256 = "2ee51c793fa8ac349e5ab41a50841c19bd311419ff8e04dbf242d38c042c30e8"
MAX_CANDIDATES = 20
DIAGNOSTIC_COUNT = 8
MODULE_NAME = Path(__file__).name
SHA256_RE = re.compile(r"[a-f0-9]{64}")


class Generation5EvaluationAuthorityError(ValueError):
    """Raised when E1 candidate or private-review authority drifts."""


def _canonical_hash(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Generation5EvaluationAuthorityError("Private E1 authority is unreadable.") from exc
    if not isinstance(value, dict):
        raise Generation5EvaluationAuthorityError("Private E1 authority must be an object.")
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
        raise Generation5EvaluationAuthorityError("Repository authority is unavailable.")
    return result.stdout if binary else result.stdout.strip()


def _repository_authority() -> dict[str, Any]:
    if _git(["status", "--porcelain=v1", "--untracked-files=normal"]):
        raise Generation5EvaluationAuthorityError("Repository must be clean.")
    parity = str(_git(["rev-list", "--left-right", "--count", "@{upstream}...HEAD"])).split()
    if parity != ["0", "0"]:
        raise Generation5EvaluationAuthorityError("Repository must be upstream-even.")
    commit = str(_git(["rev-parse", "HEAD"]))
    body = _git(["show", f"{commit}:{MODULE_NAME}"], binary=True)
    if (
        not re.fullmatch(r"[a-f0-9]{40}", commit)
        or not isinstance(body, bytes)
        or hashlib.sha256(body).hexdigest() != sha256_file(Path(__file__).resolve())
    ):
        raise Generation5EvaluationAuthorityError("Committed E1 module drifted.")
    return {
        "commit": commit,
        "module_sha256": hashlib.sha256(body).hexdigest(),
        "clean": True,
        "upstream_ahead": 0,
        "upstream_behind": 0,
    }


def _j2_authority() -> dict[str, Any]:
    replay = j2.replay_generation5_recovery_j2(J2_PREVIEW_SHA256)
    paths = j2._paths(j2.DEFAULT_RUNTIME_ROOT, J2_PREVIEW_SHA256)
    if (
        replay.get("idempotent_replay") is not True
        or sha256_file(paths["manifest"]) != J2_MANIFEST_SHA256
        or replay.get("action_vector", {}).get("enumerate_e1_candidates") is not True
        or replay.get("action_vector", {}).get("run_models_or_predictions") is not False
    ):
        raise Generation5EvaluationAuthorityError("J2 did not authorize exact E1 work.")
    return {
        "j2_preview_sha256": J2_PREVIEW_SHA256,
        "j2_manifest_sha256": J2_MANIFEST_SHA256,
        "idempotent_replay": True,
    }


def _candidate_rows() -> list[dict[str, Any]]:
    paths = r0._paths(r0.DEFAULT_RUNTIME_ROOT, R0_PREVIEW_SHA256)
    require_private_file(paths["manifest"], paths["root"])
    manifest = _read_json(paths["manifest"])
    preview = manifest.get("preview")
    if (
        sha256_file(paths["manifest"]) != R0_MANIFEST_SHA256
        or not isinstance(preview, Mapping)
        or preview.get("content_sha256") != R0_PREVIEW_SHA256
        or preview.get("selected_membership_sha256") != R0_SELECTED_MEMBERSHIP_SHA256
    ):
        raise Generation5EvaluationAuthorityError("Frozen R0 inventory drifted.")
    private = preview.get("private_evidence")
    inventory = private.get("inventory") if isinstance(private, Mapping) else None
    if not isinstance(inventory, list):
        raise Generation5EvaluationAuthorityError("Frozen R0 inventory is missing.")
    eligible = [dict(row) for row in inventory if isinstance(row, Mapping) and row.get("status") == "eligible"]
    ordered = sorted(
        eligible,
        key=lambda row: (
            str(row.get("recording_start_utc") or ""),
            str(row.get("source_sha256") or ""),
            str(row.get("transcript_sha256") or ""),
        ),
    )
    if ordered != eligible or len(ordered) <= DIAGNOSTIC_COUNT:
        raise Generation5EvaluationAuthorityError("Frozen eligible ordering is invalid.")
    return ordered[DIAGNOSTIC_COUNT : DIAGNOSTIC_COUNT + MAX_CANDIDATES]


def _speaker_cards(row: Mapping[str, Any], ordinal: int) -> list[dict[str, Any]]:
    source = Path(str(row.get("path") or "")).expanduser().absolute()
    transcript_path = Path(str(row.get("transcript_path") or "")).expanduser().absolute()
    source_hash = str(row.get("source_sha256") or "")
    transcript_hash = str(row.get("transcript_sha256") or "")
    if (
        not source.is_file()
        or source.is_symlink()
        or not transcript_path.is_file()
        or transcript_path.is_symlink()
        or not SHA256_RE.fullmatch(source_hash)
        or not SHA256_RE.fullmatch(transcript_hash)
        or sha256_file(source) != source_hash
        or sha256_file(transcript_path) != transcript_hash
    ):
        raise Generation5EvaluationAuthorityError("E1 source or transcript drifted.")
    transcript = _read_json(transcript_path)
    utterances = transcript.get("utterances")
    if not isinstance(utterances, list):
        raise Generation5EvaluationAuthorityError("E1 transcript utterances are unavailable.")
    labels = sorted(
        {
            str(item.get("speaker") or "").strip()
            for item in utterances
            if isinstance(item, Mapping) and str(item.get("speaker") or "").strip()
        }
    )
    if not labels:
        raise Generation5EvaluationAuthorityError("candidate_has_no_usable_speaker_utterance")
    case_id = "g5-case-" + _canonical_hash({"source": source_hash, "transcript": transcript_hash})[:20]
    cards = []
    for label in labels:
        candidates = []
        for index, raw in enumerate(utterances):
            if not isinstance(raw, Mapping) or str(raw.get("speaker") or "").strip() != label:
                continue
            start, end = raw.get("start"), raw.get("end")
            clue = " ".join(str(raw.get("text") or "").split())
            if isinstance(start, int) and isinstance(end, int) and end > start and clue:
                candidates.append(
                    {
                        "utterance_index": index,
                        "start_milliseconds": start,
                        "end_milliseconds": end,
                        "text": clue[:700],
                        "rank": (min(end - start, 20_000), len(clue), -index),
                    }
                )
        if not candidates:
            continue
        ranked = sorted(candidates, key=lambda item: item["rank"], reverse=True)
        best = ranked[0]
        start_seconds = max(0.0, best["start_milliseconds"] / 1000 - 1.5)
        end_seconds = min(best["end_milliseconds"] / 1000 + 1.5, start_seconds + 25.0)
        cards.append(
            {
                "case_id": case_id,
                "display_case": f"Candidate {ordinal}",
                "speaker_label": label,
                "speaker_ref": f"Candidate {ordinal} / Speaker {label}",
                "source_path": str(source),
                "transcript_path": str(transcript_path),
                "source_sha256": source_hash,
                "transcript_sha256": transcript_hash,
                "recording_start_utc": str(row.get("recording_start_utc") or ""),
                "clip": {
                    "start_seconds": round(start_seconds, 3),
                    "duration_seconds": round(end_seconds - start_seconds, 3),
                    "snippets": [
                        {key: item[key] for key in ("utterance_index", "start_milliseconds", "end_milliseconds", "text")}
                        for item in ranked[:3]
                    ],
                },
            }
        )
    if not cards:
        raise Generation5EvaluationAuthorityError("candidate_has_no_usable_speaker_utterance")
    return cards


def _tool_identity() -> dict[str, str]:
    path = shutil.which("ffmpeg")
    if not path:
        raise Generation5EvaluationAuthorityError("ffmpeg is unavailable.")
    result = subprocess.run([path, "-version"], capture_output=True, text=True, check=False)
    if result.returncode or not result.stdout.splitlines():
        raise Generation5EvaluationAuthorityError("ffmpeg identity is unavailable.")
    return {"ffmpeg_path": path, "ffmpeg_revision": result.stdout.splitlines()[0]}


def preview_generation5_evaluation_authority(
    *,
    candidate_rows: Sequence[Mapping[str, Any]] | None = None,
    j2_authority: Mapping[str, Any] | None = None,
    repository_authority: Mapping[str, Any] | None = None,
    tool_identity: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    rows = [dict(row) for row in (candidate_rows if candidate_rows is not None else _candidate_rows())]
    if not 1 <= len(rows) <= MAX_CANDIDATES:
        raise Generation5EvaluationAuthorityError("E1 candidate count is outside authority.")
    enumerated_source_hashes = [str(row.get("source_sha256") or "") for row in rows]
    enumerated_transcript_hashes = [str(row.get("transcript_sha256") or "") for row in rows]
    if len(set(enumerated_source_hashes)) != len(rows) or len(set(enumerated_transcript_hashes)) != len(rows):
        raise Generation5EvaluationAuthorityError("E1 candidate membership overlaps.")
    cards = []
    reviewable_rows = []
    rejection_ledger = []
    for ordinal, row in enumerate(rows, start=1):
        try:
            row_cards = _speaker_cards(row, ordinal)
        except Generation5EvaluationAuthorityError as exc:
            if str(exc) != "candidate_has_no_usable_speaker_utterance":
                raise
            rejection_ledger.append(
                {
                    "enumerated_ordinal": ordinal,
                    "source_sha256": str(row.get("source_sha256") or ""),
                    "transcript_sha256": str(row.get("transcript_sha256") or ""),
                    "reason_code": str(exc),
                }
            )
            continue
        reviewable_rows.append((ordinal, row))
        cards.extend(row_cards)
    if len(reviewable_rows) < 7:
        raise Generation5EvaluationAuthorityError("Fewer than seven E1 candidates are reviewable.")
    source_hashes = [str(row.get("source_sha256") or "") for _, row in reviewable_rows]
    transcript_hashes = [str(row.get("transcript_sha256") or "") for _, row in reviewable_rows]
    membership = [
        {
            "enumerated_ordinal": ordinal,
            "source_sha256": str(row.get("source_sha256") or ""),
            "transcript_sha256": str(row.get("transcript_sha256") or ""),
            "recording_start_utc": str(row.get("recording_start_utc") or ""),
        }
        for ordinal, row in reviewable_rows
    ]
    private = {
        "candidate_membership": membership,
        "candidate_rejection_ledger": rejection_ledger,
        "cards": cards,
    }
    actions = {
        "materialize_private_review_clips": True,
        "request_operator_identity_review": True,
        "freeze_cohort_or_gold": False,
        "run_models_or_predictions": False,
        "reveal_gold_to_workers": False,
        "mutate_profiles_or_references": False,
        "enable_default_integration": False,
        "run_historical_reprocessing": False,
    }
    core = {
        "schema_version": PREVIEW_SCHEMA,
        "status": "private_operator_review_ready_to_materialize",
        "repository_authority": dict(repository_authority or _repository_authority()),
        "j2_authority": dict(j2_authority or _j2_authority()),
        "r0_preview_sha256": R0_PREVIEW_SHA256,
        "r0_manifest_sha256": R0_MANIFEST_SHA256,
        "selection_contract": {
            "excluded_first_diagnostic_rows": DIAGNOSTIC_COUNT,
            "maximum_candidates": MAX_CANDIDATES,
            "candidate_order": ["recording_start_utc", "source_sha256", "transcript_sha256"],
            "cohort_rule": "lexicographically_first_seven_combination_passing_all_population_gates",
        },
        "tool_identity": dict(tool_identity or _tool_identity()),
        "enumerated_candidate_count": len(rows),
        "candidate_count": len(reviewable_rows),
        "rejected_candidate_count": len(rejection_ledger),
        "candidate_rejection_reason_counts": {
            "candidate_has_no_usable_speaker_utterance": len(rejection_ledger)
        },
        "speaker_label_count": len(cards),
        "candidate_membership_sha256": _canonical_hash(membership),
        "candidate_source_set_sha256": _canonical_hash(sorted(source_hashes)),
        "candidate_transcript_set_sha256": _canonical_hash(sorted(transcript_hashes)),
        "private_evidence": private,
        "action_vector": actions,
        "contains_paths": True,
        "contains_private_membership": True,
        "contains_transcript_text": True,
        "contains_audio_excerpts": False,
        "did_inspect_transcript_utterances": True,
        "did_access_identity_gold": False,
        "did_load_or_run_models": False,
    }
    return {**core, "content_sha256": _canonical_hash(core)}


def _slug(card: Mapping[str, Any]) -> str:
    raw = f"{card.get('display_case')}-{card.get('speaker_label')}".casefold()
    return re.sub(r"[^a-z0-9]+", "-", raw).strip("-")


def _render_page(preview: Mapping[str, Any]) -> str:
    cards = []
    for card in preview["private_evidence"]["cards"]:
        snippets = "".join(
            f"<li>{html.escape(str(item['text']))}</li>" for item in card["clip"]["snippets"]
        )
        reference = html.escape(str(card["speaker_ref"]), quote=True)
        cards.append(
            f'''<section class="card"><h2>{html.escape(str(card["display_case"]))} / Speaker {html.escape(str(card["speaker_label"]))}</h2>
<audio controls preload="metadata" src="clips/{_slug(card)}.wav"></audio>
<details><summary>Transcript clues</summary><ul>{snippets}</ul></details>
<label>Identity or stable alias <input data-answer="1" data-ref="{reference}"></label>
<p class="hint">Reuse the same name for the same person. Use UNKNOWN only if you cannot tell.</p></section>'''
        )
    return f'''<!doctype html><html><head><meta charset="utf-8"><title>Generation-5 private speaker review</title>
<style>body{{font:16px system-ui;max-width:900px;margin:2rem auto;padding:0 1rem;background:#f6f7f9;color:#18202a}}.card{{background:white;padding:1rem 1.2rem;margin:1rem 0;border-radius:12px;box-shadow:0 1px 5px #0002}}audio{{width:100%}}input{{display:block;width:min(95%,34rem);padding:.55rem;margin-top:.35rem}}button{{font-size:1rem;padding:.7rem 1rem}}textarea{{display:block;width:95%;min-height:14rem;margin:.75rem 0;padding:.6rem}}.hint{{color:#59636e;font-size:.9rem}}</style></head>
<body><h1>Private Generation-5 speaker-label review</h1><p>These are the 12 fresh candidates in their frozen order. Identify every speaker; after review, the system will select the first seven-recording combination that meets the population rules. Enrolled people to look for: Chris Williams and Eric Cochran; each must appear in at least two recordings.</p>
<button id="copy" type="button">Prepare answers</button><span id="status"></span><textarea id="answers" aria-label="Copyable answer block" placeholder="Click Prepare answers after filling every identity."></textarea>{''.join(cards)}
<script>document.getElementById('copy').onclick=async()=>{{const lines=[...document.querySelectorAll('[data-answer]')].map(x=>`${{x.dataset.ref}} = ${{x.value.trim()||'UNANSWERED'}}`);const text=lines.join('\n');const box=document.getElementById('answers');box.value=text;box.focus();box.select();const status=document.getElementById('status');try{{await navigator.clipboard.writeText(text);status.textContent=' Copied—paste into chat.';}}catch(error){{status.textContent=' Copy the selected answer block below and paste it into chat.';}}}};</script></body></html>'''


def _extract(card: Mapping[str, Any], target: Path, ffmpeg_path: str) -> None:
    result = subprocess.run(
        [
            ffmpeg_path,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            str(card["clip"]["start_seconds"]),
            "-t",
            str(card["clip"]["duration_seconds"]),
            "-i",
            str(card["source_path"]),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(target),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    if result.returncode:
        raise Generation5EvaluationAuthorityError("Private E1 clip extraction failed.")


def _paths(runtime_root: Path, content_sha256: str) -> dict[str, Path]:
    root = runtime_root.expanduser().absolute()
    run = root / f"generation5-e1-review-{content_sha256[:24]}"
    return {
        "root": root,
        "run": run,
        "clips": run / "clips",
        "manifest": run / "private-manifest.json",
        "page": run / "review.html",
        "receipt": run / "receipt.json",
    }


def _portable(preview: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": RECEIPT_SCHEMA,
        "status": "awaiting_private_operator_identity_review",
        "preview_content_sha256": preview["content_sha256"],
        "candidate_count": preview["candidate_count"],
        "enumerated_candidate_count": preview["enumerated_candidate_count"],
        "rejected_candidate_count": preview["rejected_candidate_count"],
        "speaker_label_count": preview["speaker_label_count"],
        "candidate_membership_sha256": preview["candidate_membership_sha256"],
        "candidate_source_set_sha256": preview["candidate_source_set_sha256"],
        "candidate_transcript_set_sha256": preview["candidate_transcript_set_sha256"],
        "did_access_identity_gold": False,
        "did_load_or_run_models": False,
        "did_freeze_cohort_or_gold": False,
    }


def apply_generation5_evaluation_authority(
    reviewed_preview: Mapping[str, Any],
    *,
    expected_content_sha256: str,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
) -> dict[str, Any]:
    preview = preview_generation5_evaluation_authority()
    if dict(reviewed_preview) != preview or preview["content_sha256"] != expected_content_sha256:
        raise Generation5EvaluationAuthorityError("Reviewed E1 preview is stale.")
    paths = _paths(runtime_root, expected_content_sha256)
    if paths["receipt"].exists():
        return replay_generation5_evaluation_authority(expected_content_sha256, runtime_root=runtime_root)
    ensure_private_tree(paths["root"], paths["clips"])
    manifest = {"schema_version": MANIFEST_SCHEMA, "status": "frozen", "preview": preview}
    if not paths["manifest"].exists():
        write_immutable_private_json(paths["manifest"], manifest)
    elif _read_json(paths["manifest"]) != manifest:
        raise Generation5EvaluationAuthorityError("Private E1 manifest changed in place.")
    ffmpeg_path = str(preview["tool_identity"]["ffmpeg_path"])
    for card in preview["private_evidence"]["cards"]:
        target = paths["clips"] / f"{_slug(card)}.wav"
        if not target.exists():
            _extract(card, target, ffmpeg_path)
            os.chmod(target, 0o600)
    rendered = _render_page(preview)
    if not paths["page"].exists():
        paths["page"].write_text(rendered, encoding="utf-8")
        os.chmod(paths["page"], 0o600)
    elif paths["page"].read_text(encoding="utf-8") != rendered:
        raise Generation5EvaluationAuthorityError("Private E1 review page changed in place.")
    clip_hashes = sorted(sha256_file(path) for path in paths["clips"].glob("*.wav"))
    if len(clip_hashes) != preview["speaker_label_count"]:
        raise Generation5EvaluationAuthorityError("Private E1 clip denominator is incomplete.")
    receipt = {
        **_portable(preview),
        "manifest_sha256": sha256_file(paths["manifest"]),
        "review_page_sha256": sha256_file(paths["page"]),
        "clip_set_sha256": _canonical_hash(clip_hashes),
        "clip_count": len(clip_hashes),
        "mode": "0600",
    }
    write_immutable_private_json(paths["receipt"], receipt)
    return {**receipt, "idempotent_replay": False, "private_review_page_path": str(paths["page"])}


def replay_generation5_evaluation_authority(
    expected_content_sha256: str,
    *,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
) -> dict[str, Any]:
    paths = _paths(runtime_root, expected_content_sha256)
    for key in ("manifest", "page", "receipt"):
        require_private_file(paths[key], paths["root"])
    manifest = _read_json(paths["manifest"])
    receipt = _read_json(paths["receipt"])
    preview = manifest.get("preview")
    if not isinstance(preview, Mapping):
        raise Generation5EvaluationAuthorityError("Private E1 preview is missing.")
    preview = dict(preview)
    repository = preview.get("repository_authority")
    if not isinstance(repository, Mapping):
        raise Generation5EvaluationAuthorityError("Recorded E1 repository authority is missing.")
    commit = str(repository.get("commit") or "")
    body = _git(["show", f"{commit}:{MODULE_NAME}"], binary=True) if re.fullmatch(r"[a-f0-9]{40}", commit) else b""
    _j2_authority()
    for row in preview["private_evidence"]["candidate_membership"]:
        card = next(item for item in preview["private_evidence"]["cards"] if item["case_id"].endswith(_canonical_hash({"source": row["source_sha256"], "transcript": row["transcript_sha256"]})[:20]))
        if sha256_file(Path(card["source_path"])) != row["source_sha256"] or sha256_file(Path(card["transcript_path"])) != row["transcript_sha256"]:
            raise Generation5EvaluationAuthorityError("E1 source evidence drifted.")
    clip_hashes = sorted(sha256_file(path) for path in paths["clips"].glob("*.wav"))
    expected_receipt = {
        **_portable(preview),
        "manifest_sha256": sha256_file(paths["manifest"]),
        "review_page_sha256": sha256_file(paths["page"]),
        "clip_set_sha256": _canonical_hash(clip_hashes),
        "clip_count": len(clip_hashes),
        "mode": "0600",
    }
    core = {key: value for key, value in preview.items() if key != "content_sha256"}
    if (
        preview.get("content_sha256") != expected_content_sha256
        or _canonical_hash(core) != expected_content_sha256
        or manifest != {"schema_version": MANIFEST_SCHEMA, "status": "frozen", "preview": preview}
        or receipt != expected_receipt
        or paths["page"].read_text(encoding="utf-8") != _render_page(preview)
        or len(clip_hashes) != preview.get("speaker_label_count")
        or not isinstance(body, bytes)
        or hashlib.sha256(body).hexdigest() != repository.get("module_sha256")
        or _git(["merge-base", "--is-ancestor", commit, "HEAD"]) != ""
    ):
        raise Generation5EvaluationAuthorityError("Private E1 authority drifted.")
    return {
        **receipt,
        "replay_schema_version": REPLAY_SCHEMA,
        "idempotent_replay": True,
        "private_review_page_path": str(paths["page"]),
    }
