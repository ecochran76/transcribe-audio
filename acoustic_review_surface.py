from __future__ import annotations

import html
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import urlsplit


REVIEW_SURFACE_SCHEMA = "transcribe-audio.acoustic-review-surface.v1"
NON_ENROLLED_IDENTITIES = frozenset({"neither_enrolled", "unknown"})
CARD_ID_RE = re.compile(r"^[^:\s]+::SPEAKER_[1-9][0-9]*$")
SAFE_REVIEW_LABEL_RE = re.compile(r"^[^=\r\n]{1,120}$")
CONFIDENCE_BANDS = frozenset({"none", "low", "medium", "high"})


class AcousticReviewSurfaceError(ValueError):
    """Raised when a review surface could weaken identity or export bounds."""


def _normalize_review_label(value: Any) -> str:
    label = " ".join(str(value or "").split())
    if not SAFE_REVIEW_LABEL_RE.fullmatch(label):
        raise AcousticReviewSurfaceError("A review-only label is unsafe.")
    return label


def _validate_audio_url(value: Any) -> str:
    audio_url = str(value or "")
    split = urlsplit(audio_url)
    path = PurePosixPath(split.path)
    if (
        not audio_url
        or split.scheme
        or split.netloc
        or split.query
        or split.fragment
        or split.path.startswith("/")
        or "\\" in audio_url
        or ".." in path.parts
        or path.suffix.lower() != ".wav"
    ):
        raise AcousticReviewSurfaceError("Audio URLs must be relative WAV paths.")
    return audio_url


def _validate_options(
    enrolled_options: Sequence[Mapping[str, Any]],
    *,
    allowed_subject_ids: frozenset[str],
) -> tuple[list[dict[str, str]], dict[str, str]]:
    options: list[dict[str, str]] = []
    seen: set[str] = set()
    export_labels: dict[str, str] = {}
    for raw in enrolled_options:
        machine_identity = str(raw.get("machine_identity") or "")
        display_label = " ".join(str(raw.get("display_label") or "").split())
        export_identity = " ".join(str(raw.get("export_identity") or "").split())
        if (
            machine_identity not in allowed_subject_ids
            or machine_identity in seen
            or not display_label
            or len(display_label) > 120
            or not export_identity
            or "=" in export_identity
            or any(char in export_identity for char in "\r\n")
        ):
            raise AcousticReviewSurfaceError("An enrolled decision option is invalid.")
        seen.add(machine_identity)
        export_labels[machine_identity] = export_identity
        options.append(
            {
                "machine_identity": machine_identity,
                "display_label": display_label,
                "export_identity": export_identity,
            }
        )
    if seen != set(allowed_subject_ids):
        raise AcousticReviewSurfaceError("Every allowlisted subject requires one option.")
    export_labels.update(
        {
            "neither_enrolled": "Neither enrolled person",
            "unknown": "UNKNOWN",
        }
    )
    return options, export_labels


def _validate_cards(
    cards: Sequence[Mapping[str, Any]],
    *,
    allowed_subject_ids: frozenset[str],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in cards:
        card_id = str(raw.get("card_id") or "")
        speaker_ref = str(raw.get("speaker_ref") or "")
        proposal_subject_id = raw.get("proposal_subject_id")
        confidence_band = str(raw.get("confidence_band") or "")
        supporting = raw.get("supporting_unit_count")
        opposing = raw.get("opposing_unit_count")
        transcript = str(raw.get("transcript") or "")
        proposal_label = " ".join(str(raw.get("proposal_label") or "").split())
        if (
            not CARD_ID_RE.fullmatch(card_id)
            or card_id in seen
            or card_id.rsplit("::", 1)[-1] != speaker_ref
            or proposal_subject_id not in {*allowed_subject_ids, None}
            or confidence_band not in CONFIDENCE_BANDS
            or isinstance(supporting, bool)
            or not isinstance(supporting, int)
            or supporting < 0
            or isinstance(opposing, bool)
            or not isinstance(opposing, int)
            or opposing < 0
            or not proposal_label
            or len(proposal_label) > 120
            or not transcript
            or len(transcript) > 4_000
        ):
            raise AcousticReviewSurfaceError("A review card is invalid.")
        seen.add(card_id)
        normalized.append(
            {
                "card_id": card_id,
                "speaker_ref": speaker_ref,
                "proposal_label": proposal_label,
                "proposal_subject_id": proposal_subject_id,
                "confidence_band": confidence_band,
                "supporting_unit_count": supporting,
                "opposing_unit_count": opposing,
                "transcript": transcript,
                "audio_url": _validate_audio_url(raw.get("audio_url")),
            }
        )
    if not normalized:
        raise AcousticReviewSurfaceError("At least one review card is required.")
    return normalized


def build_answer_block(
    *,
    card_ids: Sequence[str],
    decisions: Mapping[str, str],
    export_labels: Mapping[str, str],
    allowed_subject_ids: frozenset[str],
    review_display_labels: Mapping[str, str] | None = None,
) -> str:
    """Build the exact strict-importer answer block without inferring identity."""

    ordered = tuple(str(card_id) for card_id in card_ids)
    if (
        not ordered
        or len(ordered) != len(set(ordered))
        or any(not CARD_ID_RE.fullmatch(card_id) for card_id in ordered)
        or set(decisions) != set(ordered)
    ):
        raise AcousticReviewSurfaceError("Every exact card requires one decision.")
    labels = dict(review_display_labels or {})
    if set(labels) - set(ordered):
        raise AcousticReviewSurfaceError("A review-only label references an unknown card.")
    allowed = set(allowed_subject_ids) | set(NON_ENROLLED_IDENTITIES)
    rows: list[str] = []
    for card_id in ordered:
        decision = str(decisions[card_id])
        export_identity = str(export_labels.get(decision) or "")
        if decision not in allowed or not export_identity:
            raise AcousticReviewSurfaceError("A decision is not allowlisted.")
        label = labels.get(card_id)
        if label is not None:
            if decision != "neither_enrolled":
                raise AcousticReviewSurfaceError(
                    "Only neither-enrolled decisions may carry a display label."
                )
            export_identity = (
                f"Neither enrolled person ({_normalize_review_label(label)})"
            )
        rows.append(f"{card_id}={export_identity}")
    return "\n".join(rows)


def _json_for_script(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).replace(
        "<", "\\u003c"
    )


def render_review_surface(
    *,
    title: str,
    cards: Sequence[Mapping[str, Any]],
    enrolled_options: Sequence[Mapping[str, Any]],
    allowed_subject_ids: frozenset[str],
) -> str:
    """Render a deterministic no-apply review form with lazy audio."""

    normalized_title = " ".join(str(title or "").split())
    if not normalized_title or len(normalized_title) > 160:
        raise AcousticReviewSurfaceError("The review title is invalid.")
    normalized_cards = _validate_cards(
        cards, allowed_subject_ids=allowed_subject_ids
    )
    normalized_options, export_labels = _validate_options(
        enrolled_options, allowed_subject_ids=allowed_subject_ids
    )
    option_rows = [
        *normalized_options,
        {
            "machine_identity": "neither_enrolled",
            "display_label": "Neither enrolled subject",
            "export_identity": "Neither enrolled person",
        },
        {
            "machine_identity": "unknown",
            "display_label": "Unknown / cannot determine",
            "export_identity": "UNKNOWN",
        },
    ]
    sections: list[str] = []
    for ordinal, card in enumerate(normalized_cards, start=1):
        choices = []
        for option_ordinal, option in enumerate(option_rows, start=1):
            control_id = f"decision-{ordinal}-{option_ordinal}"
            choices.append(
                '<div class="choice">'
                f'<input type="radio" id="{control_id}" '
                f'name="decision-{ordinal}" '
                f'value="{html.escape(option["machine_identity"], quote=True)}" '
                f'data-card-id="{html.escape(card["card_id"], quote=True)}">'
                f'<label for="{control_id}">{html.escape(option["display_label"])}</label>'
                "</div>"
            )
        audio_url = html.escape(card["audio_url"], quote=True)
        card_id = html.escape(card["card_id"], quote=True)
        sections.append(
            f'<article class="card" data-review-card data-card-id="{card_id}">'
            f'<h2>Card {ordinal}: {html.escape(card["speaker_ref"])}</h2>'
            f'<p><strong>Proposal:</strong> {html.escape(card["proposal_label"])}</p>'
            f'<p>{html.escape(card["confidence_band"])} confidence; '
            f'{card["supporting_unit_count"]} supporting units; '
            f'{card["opposing_unit_count"]} opposing units.</p>'
            f'<audio controls preload="none" data-review-audio data-card-id="{card_id}">'
            f'<source src="{audio_url}" type="audio/wav"></audio>'
            '<div class="media-line">'
            f'<a class="audio-fallback" href="{audio_url}" target="_blank" rel="noopener">'
            "Open audio directly</a>"
            '<output class="media-status" aria-live="polite">Audio loads on demand.</output>'
            "</div>"
            f'<p class="transcript">{html.escape(card["transcript"])}</p>'
            f'<fieldset data-decision-group data-card-id="{card_id}">'
            f'<legend>Identity decision for {html.escape(card["speaker_ref"])}</legend>'
            + "".join(choices)
            + f'<label class="review-label" for="review-label-{ordinal}">Optional review-only label for neither enrolled subject</label>'
            f'<input id="review-label-{ordinal}" data-review-label data-card-id="{card_id}" '
            'type="text" maxlength="120" disabled autocomplete="off">'
            "</fieldset>"
            f'<code>{html.escape(card["card_id"])}</code>'
            "</article>"
        )
    card_order = [card["card_id"] for card in normalized_cards]
    script = f"""
const cardOrder = {_json_for_script(card_order)};
const exportLabels = {_json_for_script(export_labels)};
const form = document.querySelector('#review-form');
const answerBlock = document.querySelector('#answer-block');
const exportStatus = document.querySelector('#export-status');
const downloadLink = document.querySelector('#download-answers');

for (const audio of document.querySelectorAll('[data-review-audio]')) {{
  const status = audio.closest('[data-review-card]').querySelector('.media-status');
  audio.addEventListener('loadstart', () => {{ status.textContent = 'Loading audio...'; }});
  audio.addEventListener('loadedmetadata', () => {{ status.textContent = 'Audio ready.'; }});
  audio.addEventListener('error', () => {{ status.textContent = 'Audio failed. Use the direct-file fallback.'; }});
}}

form.addEventListener('change', (event) => {{
  if (!event.target.matches('input[type="radio"]')) return;
  const group = event.target.closest('[data-decision-group]');
  const label = group.querySelector('[data-review-label]');
  label.disabled = event.target.value !== 'neither_enrolled';
  if (label.disabled) label.value = '';
}});

function prepareAnswers() {{
  const rows = [];
  const incomplete = [];
  for (let index = 0; index < cardOrder.length; index += 1) {{
    const cardId = cardOrder[index];
    const selected = form.querySelector(`input[name="decision-${{index + 1}}"]:checked`);
    if (!selected || selected.dataset.cardId !== cardId || !exportLabels[selected.value]) {{
      incomplete.push(cardId);
      continue;
    }}
    let exported = exportLabels[selected.value];
    const group = selected.closest('[data-decision-group]');
    const reviewLabel = group.querySelector('[data-review-label]').value.trim().replace(/\\s+/g, ' ');
    if (reviewLabel) {{
      if (selected.value !== 'neither_enrolled' || reviewLabel.length > 120 || /[=\\r\\n]/.test(reviewLabel)) {{
        exportStatus.textContent = `Unsafe review-only label on ${{cardId}}.`;
        answerBlock.value = '';
        return;
      }}
      exported = `Neither enrolled person (${{reviewLabel}})`;
    }}
    rows.push(`${{cardId}}=${{exported}}`);
  }}
  if (incomplete.length) {{
    exportStatus.textContent = `${{incomplete.length}} card(s) still require an explicit decision.`;
    answerBlock.value = '';
    return;
  }}
  answerBlock.value = rows.join('\\n');
  answerBlock.focus();
  answerBlock.select();
  exportStatus.textContent = `Prepared ${{rows.length}} ordered decisions.`;
  const blob = new Blob([answerBlock.value + '\\n'], {{type: 'text/plain'}});
  if (downloadLink.dataset.objectUrl) URL.revokeObjectURL(downloadLink.dataset.objectUrl);
  downloadLink.dataset.objectUrl = URL.createObjectURL(blob);
  downloadLink.href = downloadLink.dataset.objectUrl;
  downloadLink.hidden = false;
}}

document.querySelector('#prepare-answers').addEventListener('click', prepareAnswers);
document.querySelector('#copy-answers').addEventListener('click', async () => {{
  if (!answerBlock.value) prepareAnswers();
  if (!answerBlock.value) return;
  try {{
    await navigator.clipboard.writeText(answerBlock.value);
    exportStatus.textContent = 'Copied the complete answer block.';
  }} catch (error) {{
    answerBlock.focus();
    answerBlock.select();
    exportStatus.textContent = 'Clipboard unavailable. The complete answer block is selected.';
  }}
}});
"""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="review-surface-schema" content="{REVIEW_SURFACE_SCHEMA}">
<title>{html.escape(normalized_title)}</title><style>
body{{font:16px system-ui,sans-serif;max-width:1000px;margin:2rem auto;padding:0 1rem;background:#f4f6f8;color:#17212b}}
.notice,.card,.export{{background:white;border:1px solid #ccd5df;border-radius:12px;padding:1rem;margin:1rem 0}}
.notice{{border-left:6px solid #8c5a00}}.card audio{{display:block;width:100%;margin:.8rem 0}}.transcript{{white-space:pre-wrap;color:#354554}}
fieldset{{border:1px solid #9aa9b8;border-radius:8px;margin:1rem 0;padding:.8rem}}legend{{font-weight:700}}.choice{{display:flex;gap:.55rem;margin:.55rem 0}}
input[type="radio"]{{width:1.15rem;height:1.15rem}}input[type="text"],textarea{{box-sizing:border-box;width:100%;padding:.65rem;margin:.4rem 0}}textarea{{min-height:18rem}}
button,.download{{display:inline-block;font:inherit;padding:.7rem 1rem;margin:.35rem .35rem .35rem 0}}.media-line{{display:flex;gap:1rem;flex-wrap:wrap}}.media-status{{color:#4c5d6d}}
code{{word-break:break-all}}:focus-visible{{outline:3px solid #1f6feb;outline-offset:2px}}
</style></head><body>
<main><h1>{html.escape(normalized_title)}</h1>
<div class="notice"><strong>Review evidence only.</strong> Decisions do not apply assignments, create identities, or update profiles. Audio loads only when requested; use the direct-file fallback if a browser cannot play it inline.</div>
<form id="review-form" novalidate>{''.join(sections)}
<section class="export"><h2>Prepare importer-compatible answers</h2>
<p>Every card requires one literal decision. Display labels remain review-only attributes.</p>
<button id="prepare-answers" type="button">Prepare answers</button>
<button id="copy-answers" type="button">Copy answers</button>
<a id="download-answers" class="download" download="review-answers.txt" hidden>Download answers</a>
<output id="export-status" aria-live="assertive">No answer block prepared.</output>
<label for="answer-block">Complete ordered answer block</label>
<textarea id="answer-block" readonly></textarea></section></form></main>
<script>{script}</script></body></html>"""
