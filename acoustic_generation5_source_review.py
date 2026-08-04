"""Materialize Plan 0055 S1 private speaker-review clips and HTML."""

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

import acoustic_generation5_source_expansion as s0
from acoustic_audio_derivatives import (
    ensure_private_tree,
    require_private_file,
    sha256_file,
    write_immutable_private_json,
)


PREVIEW_SCHEMA = "transcribe-audio.generation5-s1-review-preview.v1"
MANIFEST_SCHEMA = "transcribe-audio.generation5-s1-review-manifest.v1"
RECEIPT_SCHEMA = "transcribe-audio.generation5-s1-review-receipt.v1"
REPLAY_SCHEMA = "transcribe-audio.generation5-s1-review-replay.v1"
S0_PREVIEW_SHA256 = "7e2a99d8957b3e952c45454ac13fd4033f0b004e258c1700446f93a7b79c8f07"
S0_MANIFEST_SHA256 = "7e20d0e605c00cd2e5054c9713dcb6f485855ba56a83ef28822322858c6d790e"
DEFAULT_PROVIDER_ROOT = Path("~/.local/state/transcribe-audio/plan-0055/s1/provider")
DEFAULT_RUNTIME_ROOT = Path("~/.local/state/transcribe-audio/plan-0055/s1/review")
MODULE_NAME = Path(__file__).name


class Generation5SourceReviewError(ValueError):
    """Raised when S1 transcription or review evidence drifts."""


def _canonical_hash(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(body.encode()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Generation5SourceReviewError("Private S1 evidence is unreadable.") from exc
    if not isinstance(value, dict):
        raise Generation5SourceReviewError("Private S1 evidence must be an object.")
    return value


def _git(arguments: Sequence[str], *, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", *arguments], cwd=Path(__file__).resolve().parent,
        capture_output=True, text=not binary, check=False,
    )
    if result.returncode:
        raise Generation5SourceReviewError("Repository authority is unavailable.")
    return result.stdout if binary else result.stdout.strip()


def _repository_authority() -> dict[str, Any]:
    if _git(["status", "--porcelain=v1", "--untracked-files=normal"]):
        raise Generation5SourceReviewError("Repository must be clean.")
    if str(_git(["rev-list", "--left-right", "--count", "@{upstream}...HEAD"])).split() != ["0", "0"]:
        raise Generation5SourceReviewError("Repository must be upstream-even.")
    commit = str(_git(["rev-parse", "HEAD"]))
    body = _git(["show", f"{commit}:{MODULE_NAME}"], binary=True)
    if not isinstance(body, bytes) or hashlib.sha256(body).hexdigest() != sha256_file(Path(__file__).resolve()):
        raise Generation5SourceReviewError("Committed S1 module drifted.")
    return {"commit": commit, "module_sha256": hashlib.sha256(body).hexdigest(), "clean": True,
            "upstream_ahead": 0, "upstream_behind": 0}


def _source_rows() -> list[dict[str, Any]]:
    replay = s0.replay_generation5_source_expansion(S0_PREVIEW_SHA256)
    paths = s0._paths(s0.DEFAULT_RUNTIME_ROOT, S0_PREVIEW_SHA256)
    if replay.get("idempotent_replay") is not True or sha256_file(paths["manifest"]) != S0_MANIFEST_SHA256:
        raise Generation5SourceReviewError("S0 source authority drifted.")
    preview = _read_json(paths["manifest"]).get("preview")
    private = preview.get("private_evidence") if isinstance(preview, Mapping) else None
    if not isinstance(private, Mapping):
        raise Generation5SourceReviewError("S0 private membership is missing.")
    rows = [dict(row) for row in private.get("required_sources", []) + private.get("additional_candidates", [])]
    if len(rows) != 12:
        raise Generation5SourceReviewError("S0 candidate denominator drifted.")
    rows[0]["review_source_path"] = str(paths["zoom_copy"])
    for row in rows[1:]:
        row["review_source_path"] = row["path"]
    return rows


def _provider_paths(root: Path, ordinal: int, source_sha256: str) -> tuple[Path, Path]:
    selected = root.expanduser().absolute()
    if ordinal == 1:
        return selected / "required-zoom-job.json", selected / "required-zoom-transcript.json"
    prefix = f"candidate-{ordinal:02d}-{source_sha256[:16]}"
    return selected / f"{prefix}-job.json", selected / f"{prefix}-transcript.json"


def _speaker_cards(
    row: Mapping[str, Any], ordinal: int, job: Mapping[str, Any], result: Mapping[str, Any],
) -> list[dict[str, Any]]:
    source_hash = str(row.get("source_sha256") or "")
    payload = result.get("provider_payload")
    if (
        job.get("source_sha256") != source_hash
        or result.get("source_sha256") != source_hash
        or not isinstance(payload, Mapping)
        or payload.get("id") != job.get("transcript_id")
        or payload.get("status") != "completed"
    ):
        raise Generation5SourceReviewError("A provider result is not bound to its source.")
    utterances = payload.get("utterances")
    if not isinstance(utterances, list):
        raise Generation5SourceReviewError("A provider result has no diarized utterances.")
    labels = sorted({str(item.get("speaker") or "").strip() for item in utterances if isinstance(item, Mapping)} - {""})
    cards = []
    display = "Required A" if ordinal == 1 else "Required B" if ordinal == 2 else f"Candidate {ordinal}"
    for label in labels:
        candidates = []
        for index, item in enumerate(utterances):
            if not isinstance(item, Mapping) or str(item.get("speaker") or "").strip() != label:
                continue
            start, end = item.get("start"), item.get("end")
            text = " ".join(str(item.get("text") or "").split())
            if isinstance(start, int) and isinstance(end, int) and end > start and text:
                candidates.append({"utterance_index": index, "start_milliseconds": start,
                                   "end_milliseconds": end, "text": text[:700],
                                   "rank": (min(end - start, 20_000), len(text), -index)})
        if not candidates:
            raise Generation5SourceReviewError("A diarized speaker has no playable utterance.")
        ranked = sorted(candidates, key=lambda item: item["rank"], reverse=True)
        best = ranked[0]
        start_seconds = max(0.0, best["start_milliseconds"] / 1000 - 1.5)
        duration_seconds = min(25.0, best["end_milliseconds"] / 1000 + 1.5 - start_seconds)
        reference = f"{display} / Speaker {label}"
        cards.append({
            "ordinal": ordinal, "display_case": display, "speaker_label": label,
            "speaker_ref": reference, "source_sha256": source_hash,
            "source_path": str(row["review_source_path"]),
            "clip": {"start_seconds": round(start_seconds, 3),
                     "duration_seconds": round(duration_seconds, 3),
                     "snippets": [{key: item[key] for key in ("utterance_index", "start_milliseconds", "end_milliseconds", "text")}
                                  for item in ranked[:3]]},
        })
    if not cards:
        raise Generation5SourceReviewError("A candidate has no reviewable speakers.")
    return cards


def preview_generation5_source_review(
    *, rows: Sequence[Mapping[str, Any]] | None = None,
    provider_root: Path = DEFAULT_PROVIDER_ROOT,
    repository_authority: Mapping[str, Any] | None = None,
    ffmpeg_path: str | None = None,
) -> dict[str, Any]:
    source_rows = [dict(row) for row in (rows if rows is not None else _source_rows())]
    if len(source_rows) != 12:
        raise Generation5SourceReviewError("Exactly twelve frozen candidates are required.")
    ffmpeg = ffmpeg_path or shutil.which("ffmpeg")
    if not ffmpeg:
        raise Generation5SourceReviewError("ffmpeg is unavailable.")
    cards, membership, transcript_hashes = [], [], []
    for ordinal, row in enumerate(source_rows, start=1):
        source = Path(str(row.get("review_source_path") or "")).expanduser().absolute()
        source_hash = str(row.get("source_sha256") or "")
        if not source.is_file() or source.is_symlink() or sha256_file(source) != source_hash:
            raise Generation5SourceReviewError("A frozen source drifted.")
        job_path, result_path = _provider_paths(provider_root, ordinal, source_hash)
        require_private_file(job_path, provider_root.expanduser().absolute())
        require_private_file(result_path, provider_root.expanduser().absolute())
        job, result = _read_json(job_path), _read_json(result_path)
        row_cards = _speaker_cards(row, ordinal, job, result)
        result_hash = sha256_file(result_path)
        transcript_hashes.append(result_hash)
        membership.append({"ordinal": ordinal, "source_sha256": source_hash,
                           "provider_result_sha256": result_hash, "speaker_count": len(row_cards)})
        cards.extend(row_cards)
    core = {
        "schema_version": PREVIEW_SCHEMA, "status": "private_operator_review_ready_to_materialize",
        "repository_authority": dict(repository_authority or _repository_authority()),
        "s0_preview_sha256": S0_PREVIEW_SHA256, "s0_manifest_sha256": S0_MANIFEST_SHA256,
        "candidate_count": len(source_rows), "speaker_label_count": len(cards),
        "ordered_source_set_sha256": _canonical_hash([row["source_sha256"] for row in source_rows]),
        "provider_result_set_sha256": _canonical_hash(transcript_hashes),
        "private_evidence": {"membership": membership, "cards": cards},
        "tool_identity": {"ffmpeg_path": ffmpeg},
        "action_vector": {"materialize_private_review_clips": True, "request_operator_identity_review": True,
                          "access_or_freeze_gold": False, "run_acoustic_identity_models": False,
                          "mutate_profiles_or_references": False, "enable_default_integration": False,
                          "run_historical_reprocessing": False},
        "contains_paths": True, "contains_transcript_text": True,
        "did_transcribe_and_diarize": True, "did_access_identity_gold": False,
        "did_run_acoustic_identity_models": False,
    }
    return {**core, "content_sha256": _canonical_hash(core)}


def _slug(card: Mapping[str, Any]) -> str:
    raw = f"{card['ordinal']:02d}-{card['speaker_label']}".casefold()
    return re.sub(r"[^a-z0-9]+", "-", raw).strip("-")


def _render_page(preview: Mapping[str, Any]) -> str:
    sections = []
    for card in preview["private_evidence"]["cards"]:
        clues = "".join(f"<li>{html.escape(item['text'])}</li>" for item in card["clip"]["snippets"])
        ref = html.escape(card["speaker_ref"], quote=True)
        sections.append(
            f'<section class="card"><h2>{html.escape(card["speaker_ref"])}</h2>'
            f'<audio controls preload="metadata" src="clips/{_slug(card)}.wav"></audio>'
            f'<details><summary>Transcript clues</summary><ul>{clues}</ul></details>'
            f'<label>Identity or stable alias<input data-answer="1" data-ref="{ref}"></label>'
            '<p class="hint">Reuse the same identity for the same person. Use UNKNOWN only if you cannot tell.</p></section>'
        )
    return f'''<!doctype html><html><head><meta charset="utf-8"><title>Generation-5 private speaker review</title>
<style>body{{font:16px system-ui;max-width:920px;margin:2rem auto;padding:0 1rem;background:#f5f7fa;color:#18202a}}.card{{background:#fff;padding:1rem 1.2rem;margin:1rem 0;border-radius:12px;box-shadow:0 1px 5px #0002}}audio{{width:100%}}input,textarea{{display:block;width:95%;padding:.6rem;margin:.5rem 0}}textarea{{min-height:16rem}}button{{font-size:1rem;padding:.7rem 1rem}}.hint{{color:#59636e;font-size:.9rem}}</style></head><body>
<h1>Private Generation-5 speaker-label review</h1><p>Identify every speaker in the twelve frozen recordings. Required A is the Zoom recording and Required B is the Agritalk recording. Enrolled people to look for: Chris Williams and Eric Cochran; both should appear in Required A and Required B.</p>
<button id="prepare" type="button">Prepare answers</button><span id="status"></span><textarea id="answers" aria-label="Copyable answer block" placeholder="Fill every identity, then click Prepare answers."></textarea>{''.join(sections)}
<script>document.getElementById('prepare').onclick=async()=>{{const lines=[...document.querySelectorAll('[data-answer]')].map(x=>`${{x.dataset.ref}} = ${{x.value.trim()||'UNANSWERED'}}`);const text=lines.join('\n');const box=document.getElementById('answers');box.value=text;box.focus();box.select();try{{await navigator.clipboard.writeText(text);document.getElementById('status').textContent=' Copied—paste into chat.'}}catch(e){{document.getElementById('status').textContent=' Copy the selected block below and paste it into chat.'}}}};</script></body></html>'''


def _paths(runtime_root: Path, content_sha256: str) -> dict[str, Path]:
    root = runtime_root.expanduser().absolute()
    run = root / f"generation5-s1-review-{content_sha256[:24]}"
    return {"root": root, "run": run, "clips": run / "clips", "manifest": run / "private-manifest.json",
            "page": run / "review.html", "receipt": run / "receipt.json"}


def _extract(card: Mapping[str, Any], target: Path, ffmpeg_path: str) -> None:
    result = subprocess.run(
        [ffmpeg_path, "-nostdin", "-hide_banner", "-loglevel", "error", "-ss",
         str(card["clip"]["start_seconds"]), "-t", str(card["clip"]["duration_seconds"]),
         "-i", card["source_path"], "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(target)],
        capture_output=True, text=True, check=False, timeout=120,
    )
    if result.returncode or not target.is_file() or target.stat().st_size <= 44:
        raise Generation5SourceReviewError("Private S1 clip extraction failed.")
    target.chmod(0o600)


def apply_generation5_source_review(
    reviewed_preview: Mapping[str, Any], *, expected_content_sha256: str,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
) -> dict[str, Any]:
    preview = preview_generation5_source_review()
    if dict(reviewed_preview) != preview or preview["content_sha256"] != expected_content_sha256:
        raise Generation5SourceReviewError("Reviewed S1 preview is stale.")
    paths = _paths(runtime_root, expected_content_sha256)
    if paths["receipt"].exists():
        return replay_generation5_source_review(expected_content_sha256, runtime_root=runtime_root)
    ensure_private_tree(paths["root"], paths["clips"])
    write_immutable_private_json(paths["manifest"], {"schema_version": MANIFEST_SCHEMA, "status": "frozen", "preview": preview})
    for card in preview["private_evidence"]["cards"]:
        target = paths["clips"] / f"{_slug(card)}.wav"
        _extract(card, target, preview["tool_identity"]["ffmpeg_path"])
    page = _render_page(preview)
    descriptor = os.open(paths["page"], os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(page)
    clip_hashes = sorted(sha256_file(path) for path in paths["clips"].glob("*.wav"))
    receipt = {"schema_version": RECEIPT_SCHEMA, "status": "awaiting_private_operator_identity_review",
               "preview_content_sha256": expected_content_sha256, "candidate_count": preview["candidate_count"],
               "speaker_label_count": preview["speaker_label_count"], "clip_count": len(clip_hashes),
               "clip_set_sha256": _canonical_hash(clip_hashes), "review_page_sha256": sha256_file(paths["page"]),
               "manifest_sha256": sha256_file(paths["manifest"]), "did_access_identity_gold": False,
               "did_run_acoustic_identity_models": False, "mode": "0600"}
    if receipt["clip_count"] != preview["speaker_label_count"]:
        raise Generation5SourceReviewError("Private S1 clip denominator is incomplete.")
    write_immutable_private_json(paths["receipt"], receipt)
    return {**receipt, "idempotent_replay": False, "private_review_page_path": str(paths["page"])}


def replay_generation5_source_review(
    expected_content_sha256: str, *, runtime_root: Path = DEFAULT_RUNTIME_ROOT,
) -> dict[str, Any]:
    paths = _paths(runtime_root, expected_content_sha256)
    for key in ("manifest", "page", "receipt"):
        require_private_file(paths[key], paths["root"])
    manifest, receipt = _read_json(paths["manifest"]), _read_json(paths["receipt"])
    preview = manifest.get("preview")
    if not isinstance(preview, Mapping):
        raise Generation5SourceReviewError("Private S1 preview is missing.")
    clip_hashes = sorted(sha256_file(path) for path in paths["clips"].glob("*.wav"))
    if (
        preview.get("content_sha256") != expected_content_sha256
        or _canonical_hash({key: value for key, value in preview.items() if key != "content_sha256"}) != expected_content_sha256
        or paths["page"].read_text(encoding="utf-8") != _render_page(preview)
        or len(clip_hashes) != preview.get("speaker_label_count")
        or receipt.get("clip_set_sha256") != _canonical_hash(clip_hashes)
        or receipt.get("review_page_sha256") != sha256_file(paths["page"])
        or receipt.get("manifest_sha256") != sha256_file(paths["manifest"])
    ):
        raise Generation5SourceReviewError("Private S1 review authority drifted.")
    return {**receipt, "replay_schema_version": REPLAY_SCHEMA, "idempotent_replay": True,
            "private_review_page_path": str(paths["page"])}
