"""Build one private, batchable listening review for Generation-4 gold."""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import subprocess
import unicodedata
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


BUNDLE_SCHEMA = "transcribe-audio.generation4-private-gold-review-bundle.v1"
RECEIPT_SCHEMA = "transcribe-audio.generation4-private-gold-review-receipt.v1"
GOLD_SCHEMA = "transcribe-audio.generation4-private-gold-feasibility.v1"
GOLD_RECEIPT_SCHEMA = "transcribe-audio.generation4-private-gold-answer-receipt.v1"
GENERATION3_GOLD_MANIFEST_SCHEMA = "transcribe-audio.generation3-gold-manifest.v1"
_CASE_RE = re.compile(r"g1a-case-[a-f0-9]{20}")


class Generation4GoldReviewError(ValueError):
    """Raised when the private listening review cannot remain exact."""


def _canonical_hash(value: Any) -> str:
    body = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _case_id(source_sha256: str, transcript_sha256: str) -> str:
    digest = _canonical_hash(
        {"source": source_sha256, "transcript": transcript_sha256}
    )
    return f"g1a-case-{digest[:20]}"


def _read_private_json(path: Path) -> dict[str, Any]:
    selected = path.expanduser().absolute()
    if (
        not selected.is_file()
        or selected.is_symlink()
        or selected.stat().st_mode & 0o077
    ):
        raise Generation4GoldReviewError("Review input must be a private 0600 file.")
    try:
        value = json.loads(selected.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Generation4GoldReviewError("Review input is unreadable.") from exc
    if not isinstance(value, dict):
        raise Generation4GoldReviewError("Review input must be an object.")
    return value


def _read_source_json(path: Path) -> dict[str, Any]:
    """Read an existing bound source transcript without changing its mode."""
    selected = path.expanduser().absolute()
    if not selected.is_file() or selected.is_symlink():
        raise Generation4GoldReviewError("Source transcript is unavailable.")
    try:
        value = json.loads(selected.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Generation4GoldReviewError("Source transcript is unreadable.") from exc
    if not isinstance(value, dict):
        raise Generation4GoldReviewError("Source transcript must be an object.")
    return value


def _utterance_plan(transcript_path: Path, speaker_label: str) -> dict[str, Any]:
    transcript = _read_source_json(transcript_path)
    utterances = transcript.get("utterances")
    if not isinstance(utterances, list):
        raise Generation4GoldReviewError("Transcript utterances are unavailable.")
    candidates = []
    for index, value in enumerate(utterances):
        if not isinstance(value, Mapping) or str(value.get("speaker") or "") != speaker_label:
            continue
        start = value.get("start")
        end = value.get("end")
        text = " ".join(str(value.get("text") or "").split())
        if not isinstance(start, int) or not isinstance(end, int) or end <= start or not text:
            continue
        candidates.append(
            {
                "utterance_index": index,
                "start_milliseconds": start,
                "end_milliseconds": end,
                "text": text[:700],
                "rank": (min(end - start, 20_000), len(text), -index),
            }
        )
    if not candidates:
        raise Generation4GoldReviewError("A speaker label has no usable utterance.")
    ranked = sorted(candidates, key=lambda item: item["rank"], reverse=True)
    best = ranked[0]
    start_seconds = max(0.0, best["start_milliseconds"] / 1000 - 1.5)
    end_seconds = min(best["end_milliseconds"] / 1000 + 1.5, start_seconds + 25.0)
    snippets = []
    for item in ranked[:3]:
        snippets.append(
            {
                key: item[key]
                for key in (
                    "utterance_index", "start_milliseconds", "end_milliseconds", "text"
                )
            }
        )
    return {
        "start_seconds": round(start_seconds, 3),
        "duration_seconds": round(end_seconds - start_seconds, 3),
        "snippets": snippets,
    }


def build_generation4_gold_review_plan(
    *,
    rows: Sequence[Mapping[str, Any]],
    gap_packet: Mapping[str, Any],
    swap_packet: Mapping[str, Any],
    enrolled_identity_names: Sequence[str] = (),
) -> dict[str, Any]:
    """Join the opaque best subset to private transcripts and review labels."""
    row_by_case: dict[str, dict[str, Any]] = {}
    for raw in rows:
        row = dict(raw)
        case_id = _case_id(
            str(row.get("source_sha256") or ""),
            str(row.get("transcript_sha256") or ""),
        )
        row_by_case[case_id] = row
    best = swap_packet.get("opaque_best_subset_case_ids")
    if not isinstance(best, list) or not best or any(
        not isinstance(value, str) or not _CASE_RE.fullmatch(value) for value in best
    ):
        raise Generation4GoldReviewError("Opaque best-subset membership is invalid.")
    if any(case_id not in row_by_case for case_id in best):
        raise Generation4GoldReviewError("A best-subset case is missing from private membership.")

    label_by_ref = {
        str(item.get("speaker_ref") or ""): str(item.get("speaker_label") or "")
        for field in ("review_gaps", "supported_operator_assertions")
        for item in gap_packet.get(field) or []
        if isinstance(item, Mapping)
    }
    old_case_numbers: dict[str, int] = {}
    ref_by_case_label: dict[tuple[str, str], str] = {}
    cases = gap_packet.get("cases")
    if isinstance(cases, list):
        for number, raw_case in enumerate(cases, start=1):
            if not isinstance(raw_case, Mapping):
                continue
            case_id = str(raw_case.get("case_id") or "")
            old_case_numbers[case_id] = number
            for review in raw_case.get("speaker_reviews") or []:
                if not isinstance(review, Mapping):
                    continue
                ref = str(review.get("speaker_ref") or "")
                label = label_by_ref.get(ref, "")
                if label:
                    ref_by_case_label[(case_id, label)] = ref

    supported = {
        str(item.get("speaker_ref") or ""): " ".join(
            str(item.get("person_display_name") or "").split()
        )
        for item in gap_packet.get("supported_operator_assertions") or []
        if isinstance(item, Mapping)
    }
    replacement_ids = sorted(case_id for case_id in best if case_id not in old_case_numbers)
    replacement_names = {
        case_id: f"Replacement {chr(65 + index)}"
        for index, case_id in enumerate(replacement_ids)
    }
    ordered_ids = sorted(
        best,
        key=lambda case_id: (
            0 if case_id in old_case_numbers else 1,
            old_case_numbers.get(case_id, 0),
            case_id,
        ),
    )
    cards = []
    for case_id in ordered_ids:
        row = row_by_case[case_id]
        display_case = (
            f"Case {old_case_numbers[case_id]}"
            if case_id in old_case_numbers
            else replacement_names[case_id]
        )
        source_path = Path(str(row.get("source_path") or "")).expanduser().absolute()
        transcript_path = Path(str(row.get("transcript_path") or "")).expanduser().absolute()
        if not source_path.is_file() or source_path.is_symlink():
            raise Generation4GoldReviewError("Review source audio is unavailable.")
        labels = row.get("speaker_labels")
        if not isinstance(labels, list) or not labels:
            raise Generation4GoldReviewError("Review speaker labels are unavailable.")
        for label in sorted(str(value) for value in labels):
            ref = ref_by_case_label.get((case_id, label), f"{replacement_names.get(case_id, display_case)}:{label}")
            cards.append(
                {
                    "case_id": case_id,
                    "display_case": display_case,
                    "speaker_label": label,
                    "speaker_ref": ref,
                    "source_path": str(source_path),
                    "transcript_path": str(transcript_path),
                    "source_sha256": str(row.get("source_sha256") or ""),
                    "transcript_sha256": str(row.get("transcript_sha256") or ""),
                    "prefilled_name": supported.get(ref, ""),
                    "clip": _utterance_plan(transcript_path, label),
                }
            )
    enrolled_names = sorted(
        {" ".join(str(value or "").split()) for value in enrolled_identity_names}
        - {""},
        key=str.casefold,
    )
    if enrolled_names and len(enrolled_names) != 2:
        raise Generation4GoldReviewError(
            "Exactly two enrolled identity names are required when provided."
        )
    core = {
        "schema_version": BUNDLE_SCHEMA,
        "status": "private_operator_review_ready",
        "case_count": len(ordered_ids),
        "speaker_label_count": len(cards),
        "manual_label_count": sum(not card["prefilled_name"] for card in cards),
        "enrolled_identity_names": enrolled_names,
        "cards": cards,
        "contains_paths": True,
        "contains_private_membership": True,
        "contains_transcript_text": True,
        "contains_audio_excerpts": False,
        "contains_acoustic_scores": False,
        "did_run_acoustic_models": False,
        "did_freeze_cohort_or_gold": False,
        "supplemental_media_consumed": False,
    }
    return {**core, "content_sha256": _canonical_hash(core)}


def _ffmpeg_extract(
    source: Path, start_seconds: float, duration_seconds: float, target: Path
) -> None:
    result = subprocess.run(
        [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-ss", str(start_seconds), "-t", str(duration_seconds), "-i", str(source),
            "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(target),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode:
        raise Generation4GoldReviewError("Private review clip extraction failed.")


def _slug(display_case: str, label: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", f"{display_case}-{label}".casefold()).strip("-")
    return value or hashlib.sha256(f"{display_case}:{label}".encode()).hexdigest()[:16]


def _render_page(plan: Mapping[str, Any]) -> str:
    cards = []
    for card in plan["cards"]:
        slug = _slug(str(card["display_case"]), str(card["speaker_label"]))
        snippets = "".join(
            f"<li>{html.escape(str(item['text']))}</li>"
            for item in card["clip"]["snippets"]
        )
        value = html.escape(str(card.get("prefilled_name") or ""), quote=True)
        locked = " readonly" if value else ""
        cards.append(
            f"""<section class="card"><h2>{html.escape(str(card['display_case']))} / Speaker {html.escape(str(card['speaker_label']))}</h2>
<audio controls preload="metadata" src="clips/{slug}.wav"></audio>
<details><summary>Transcript clues</summary><ul>{snippets}</ul></details>
<p><code>{html.escape(str(card['display_case']))} / Speaker {html.escape(str(card['speaker_label']))} = ...</code></p>
<label>Identity or stable alias <input data-answer="1" data-ref="{html.escape(str(card['speaker_ref']), quote=True)}" value="{value}"{locked}></label>
<p class="hint">Use the same name or alias in multiple cards if it is the same person. Use UNKNOWN only if you cannot tell.</p></section>"""
        )
    enrolled_names = [
        html.escape(str(value)) for value in plan.get("enrolled_identity_names") or []
    ]
    enrolled_notice = (
        "<aside><strong>Enrolled people to look for:</strong> "
        + ", ".join(enrolled_names)
        + ". Both must be identified in at least two different recordings.</aside>"
        if enrolled_names
        else ""
    )
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Generation-4 private speaker review</title>
<style>body{{font:16px system-ui;max-width:900px;margin:2rem auto;padding:0 1rem;background:#f6f7f9;color:#18202a}}.card{{background:white;padding:1rem 1.2rem;margin:1rem 0;border-radius:12px;box-shadow:0 1px 5px #0002}}audio{{width:100%}}input{{display:block;width:min(95%,34rem);padding:.55rem;margin-top:.35rem}}button{{font-size:1rem;padding:.7rem 1rem}}.hint{{color:#59636e;font-size:.9rem}}</style></head>
<body><h1>Private speaker-label review</h1><p>Listen to each clip and identify the speaker. Reuse an identity or stable alias when the same person appears again. Prefilled answers came from your prior operator statements.</p>{enrolled_notice}
<button id="copy">Copy all answers</button><span id="status"></span>{''.join(cards)}
<script>document.getElementById('copy').onclick=async()=>{{const lines=[...document.querySelectorAll('[data-answer]')].map(x=>`${{x.dataset.ref}} = ${{x.value.trim()||'UNANSWERED'}}`);await navigator.clipboard.writeText(lines.join('\n'));document.getElementById('status').textContent=' Copied—paste into chat.';}};</script></body></html>"""


def apply_generation4_gold_review_bundle(
    plan: Mapping[str, Any], *, output_root: Path,
    extractor: Callable[[Path, float, float, Path], None] = _ffmpeg_extract,
) -> dict[str, Any]:
    """Materialize private clips and a single local review page."""
    if plan.get("schema_version") != BUNDLE_SCHEMA:
        raise Generation4GoldReviewError("Review plan schema is invalid.")
    expected = dict(plan)
    content_sha = str(expected.pop("content_sha256", ""))
    if content_sha != _canonical_hash(expected):
        raise Generation4GoldReviewError("Review plan content hash drifted.")
    root = output_root.expanduser().absolute()
    run = root / f"generation4-gold-review-{content_sha[:24]}"
    clips = run / "clips"
    with _private_dirs(root, run, clips):
        pass
    for card in plan["cards"]:
        target = clips / f"{_slug(str(card['display_case']), str(card['speaker_label']))}.wav"
        if not target.exists():
            extractor(
                Path(str(card["source_path"])),
                float(card["clip"]["start_seconds"]),
                float(card["clip"]["duration_seconds"]),
                target,
            )
            os.chmod(target, 0o600)
    page = run / "review.html"
    rendered = _render_page(plan)
    if page.exists() and page.read_text(encoding="utf-8") != rendered:
        raise Generation4GoldReviewError("Private review page drifted.")
    if not page.exists():
        page.write_text(rendered, encoding="utf-8")
        os.chmod(page, 0o600)
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "private_operator_review_ready",
        "plan_content_sha256": content_sha,
        "case_count": plan["case_count"],
        "speaker_label_count": plan["speaker_label_count"],
        "manual_label_count": plan["manual_label_count"],
        "private_review_page_path": str(page),
        "contains_paths": True,
        "contains_private_membership": False,
        "contains_transcript_text": True,
        "contains_audio_excerpts": True,
        "contains_acoustic_scores": False,
        "did_run_acoustic_models": False,
        "did_freeze_cohort_or_gold": False,
        "supplemental_media_consumed": False,
    }
    receipt_path = run / "receipt.json"
    body = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if receipt_path.exists() and receipt_path.read_text(encoding="utf-8") != body:
        raise Generation4GoldReviewError("Private review receipt drifted.")
    if not receipt_path.exists():
        receipt_path.write_text(body, encoding="utf-8")
        os.chmod(receipt_path, 0o600)
    return receipt


class _private_dirs:
    def __init__(self, *paths: Path):
        self.paths = paths
        self.prior = 0

    def __enter__(self):
        self.prior = os.umask(0o077)
        for path in self.paths:
            path.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(path, 0o700)
        return self

    def __exit__(self, exc_type, exc, traceback):
        os.umask(self.prior)
        return False


def _normalized_identity(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return " ".join(text.split()).casefold()


def _validated_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(plan)
    claimed = str(value.pop("content_sha256", ""))
    if (
        plan.get("schema_version") != BUNDLE_SCHEMA
        or claimed != _canonical_hash(value)
        or plan.get("did_run_acoustic_models") is not False
        or plan.get("did_freeze_cohort_or_gold") is not False
        or plan.get("supplemental_media_consumed") is not False
    ):
        raise Generation4GoldReviewError("Private review plan authority drifted.")
    return dict(plan)


def parse_generation4_gold_review_answers(
    answer_text: str, *, expected_refs: Sequence[str]
) -> dict[str, str]:
    """Parse the page's copied answer block without accepting partial gold."""
    expected = list(expected_refs)
    if not expected or len(expected) != len(set(expected)):
        raise Generation4GoldReviewError("Expected speaker references are invalid.")
    answers: dict[str, str] = {}
    for raw_line in str(answer_text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "=" not in line:
            raise Generation4GoldReviewError("Every answer line must contain '='.")
        reference, raw_value = line.split("=", 1)
        reference = reference.strip()
        value = " ".join(raw_value.split())
        if reference not in expected or reference in answers:
            raise Generation4GoldReviewError("Answer references are unknown or duplicated.")
        if (
            not value
            or _normalized_identity(value) in {"unknown", "unanswered", "not sure", "unsure"}
            or len(value) > 200
            or any(ord(character) < 32 for character in value)
        ):
            raise Generation4GoldReviewError("Every speaker needs a stable identity or alias.")
        answers[reference] = value
    if set(answers) != set(expected):
        raise Generation4GoldReviewError("The private speaker review is incomplete.")
    return answers


def _opaque_person_id(identity: str) -> str:
    digest = hashlib.sha256(_normalized_identity(identity).encode("utf-8")).hexdigest()
    return f"g4-person-{digest[:24]}"


def _opaque_conversation_id(case_id: str) -> str:
    return f"g4-conversation-{hashlib.sha256(case_id.encode()).hexdigest()[:24]}"


def _opaque_recording_id(source_sha256: str) -> str:
    return f"g4-recording-{hashlib.sha256(source_sha256.encode()).hexdigest()[:24]}"


def load_generation3_enrolled_identity_map(
    manifest_path: Path, *, expected_manifest_sha256: str
) -> dict[str, str]:
    """Load the two pre-existing enrolled names from immutable private gold."""
    path = manifest_path.expanduser().absolute()
    if not re.fullmatch(r"[a-f0-9]{64}", expected_manifest_sha256):
        raise Generation4GoldReviewError("Generation-3 gold hash is invalid.")
    manifest = _read_private_json(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_manifest_sha256:
        raise Generation4GoldReviewError("Generation-3 gold manifest drifted.")
    preview = manifest.get("preview")
    bindings = (
        preview.get("enrolled_identity_bindings")
        if isinstance(preview, Mapping)
        else None
    )
    if (
        manifest.get("schema_version") != GENERATION3_GOLD_MANIFEST_SCHEMA
        or manifest.get("status") != "applied_gold_frozen_evaluation_not_revealed"
        or not isinstance(bindings, list)
        or len(bindings) != 2
    ):
        raise Generation4GoldReviewError("Generation-3 enrolled identity authority is invalid.")
    result: dict[str, str] = {}
    for binding in bindings:
        if not isinstance(binding, Mapping):
            raise Generation4GoldReviewError("Generation-3 enrolled binding is invalid.")
        name = " ".join(str(binding.get("identity_name") or "").split())
        normalized = _normalized_identity(name)
        person_ref = str(binding.get("person_ref_id") or "")
        name_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        if (
            not normalized
            or binding.get("identity_name_sha256") != name_hash
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{2,127}", person_ref)
            or normalized in result
        ):
            raise Generation4GoldReviewError("Generation-3 enrolled binding drifted.")
        result[name] = person_ref
    return result


def build_generation4_private_gold(
    *,
    plan: Mapping[str, Any],
    answer_text: str,
    enrolled_identity_map: Mapping[str, str],
) -> dict[str, Any]:
    """Turn one complete copied review into the private G1A GOLD_SCHEMA packet."""
    authority = _validated_plan(plan)
    cards = authority.get("cards")
    if not isinstance(cards, list) or not cards:
        raise Generation4GoldReviewError("Private review cards are unavailable.")
    expected_refs = [str(card.get("speaker_ref") or "") for card in cards]
    answers = parse_generation4_gold_review_answers(
        answer_text, expected_refs=expected_refs
    )
    enrolled: dict[str, str] = {}
    for raw_identity, raw_subject in enrolled_identity_map.items():
        identity = _normalized_identity(raw_identity)
        subject = str(raw_subject or "").strip()
        if not identity or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{2,127}", subject):
            raise Generation4GoldReviewError("Enrolled identity binding is invalid.")
        if identity in enrolled and enrolled[identity] != subject:
            raise Generation4GoldReviewError("Enrolled identity binding is ambiguous.")
        enrolled[identity] = subject

    cases_by_id: dict[str, dict[str, Any]] = {}
    answer_projection = []
    for raw_card in cards:
        if not isinstance(raw_card, Mapping):
            raise Generation4GoldReviewError("Private review card is invalid.")
        card = dict(raw_card)
        case_id = str(card.get("case_id") or "")
        reference = str(card.get("speaker_ref") or "")
        label = str(card.get("speaker_label") or "")
        source_sha = str(card.get("source_sha256") or "")
        transcript_sha = str(card.get("transcript_sha256") or "")
        if (
            not _CASE_RE.fullmatch(case_id)
            or not label
            or not re.fullmatch(r"[a-f0-9]{64}", source_sha)
            or not re.fullmatch(r"[a-f0-9]{64}", transcript_sha)
            or _case_id(source_sha, transcript_sha) != case_id
        ):
            raise Generation4GoldReviewError("Private review membership drifted.")
        identity = answers[reference]
        normalized = _normalized_identity(identity)
        case = cases_by_id.setdefault(
            case_id,
            {
                "case_id": case_id,
                "source_sha256": source_sha,
                "transcript_sha256": transcript_sha,
                "conversation_id": _opaque_conversation_id(case_id),
                "recording_id": _opaque_recording_id(source_sha),
                "speaker_gold": [],
                "overlap_codes": [],
            },
        )
        if case["source_sha256"] != source_sha or case["transcript_sha256"] != transcript_sha:
            raise Generation4GoldReviewError("A review case has conflicting membership.")
        case["speaker_gold"].append(
            {
                "speaker_label": label,
                "person_id": _opaque_person_id(identity),
                "enrolled_subject_id": enrolled.get(normalized, ""),
                "private_identity_display": identity,
                "operator_answer_sha256": hashlib.sha256(
                    f"{reference}={identity}".encode("utf-8")
                ).hexdigest(),
            }
        )
        answer_projection.append(
            {"speaker_ref": reference, "identity_sha256": hashlib.sha256(normalized.encode()).hexdigest()}
        )
    cases = sorted(cases_by_id.values(), key=lambda item: item["case_id"])
    if len(cases) != authority.get("case_count"):
        raise Generation4GoldReviewError("Private review case count drifted.")
    core = {
        "schema_version": GOLD_SCHEMA,
        "status": "private_operator_review_complete_not_frozen",
        "review_plan_sha256": authority["content_sha256"],
        "answer_set_sha256": _canonical_hash(sorted(answer_projection, key=lambda item: item["speaker_ref"])),
        "cases": cases,
        "contains_paths": False,
        "contains_private_membership": True,
        "contains_identity_names_or_aliases": True,
        "contains_transcript_text": False,
        "contains_audio": False,
        "contains_acoustic_scores": False,
        "did_run_acoustic_models": False,
        "did_reveal_gold_to_prediction_workers": False,
        "did_freeze_cohort_or_gold": False,
        "supplemental_media_consumed": False,
    }
    return {**core, "content_sha256": _canonical_hash(core)}


def apply_generation4_private_gold(
    packet: Mapping[str, Any], *, output_root: Path
) -> dict[str, Any]:
    """Persist a complete but still-unfrozen private gold answer packet."""
    value = dict(packet)
    claimed = str(value.pop("content_sha256", ""))
    if (
        packet.get("schema_version") != GOLD_SCHEMA
        or claimed != _canonical_hash(value)
        or packet.get("did_run_acoustic_models") is not False
        or packet.get("did_reveal_gold_to_prediction_workers") is not False
        or packet.get("did_freeze_cohort_or_gold") is not False
    ):
        raise Generation4GoldReviewError("Private gold answer packet drifted.")
    root = output_root.expanduser().absolute()
    run = root / f"generation4-private-gold-{claimed[:24]}"
    with _private_dirs(root, run):
        pass
    path = run / "private-gold.json"
    body = json.dumps(dict(packet), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != body:
        raise Generation4GoldReviewError("Private gold answer packet changed in place.")
    if not path.exists():
        path.write_text(body, encoding="utf-8")
        os.chmod(path, 0o600)
    return {
        "schema_version": GOLD_RECEIPT_SCHEMA,
        "status": "private_operator_review_complete_not_frozen",
        "private_gold_path": str(path),
        "private_gold_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "content_sha256": claimed,
        "case_count": len(packet.get("cases") or []),
        "contains_paths": True,
        "contains_private_membership": False,
        "did_run_acoustic_models": False,
        "did_reveal_gold_to_prediction_workers": False,
        "did_freeze_cohort_or_gold": False,
        "supplemental_media_consumed": False,
        "mode": "0600",
    }
