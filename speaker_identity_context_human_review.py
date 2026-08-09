"""Build a private, direct-audio Plan 0062 human review worksheet."""

from __future__ import annotations

import html
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Mapping

from acoustic_audio_derivatives import (
    canonical_artifact_hash,
    ensure_private_tree,
    read_private_object,
    require_private_file,
    sha256_file,
    write_immutable_private_json,
)
from speaker_identity_orchestration import AcousticEvidenceBundle, negative_action_vector
from speaker_identity_plan0062_execution import (
    EXPECTED_DOCUMENTS,
    EXPECTED_SPEAKER_COUNTS,
    MANIFEST_SCHEMA as P3_MANIFEST_SCHEMA,
)


PACKET_SCHEMA = "transcribe-audio.plan0062-human-review-packet.v1"
WORKSHEET_SCHEMA = "transcribe-audio.plan0062-human-review-worksheet.v1"
SUBMISSION_SCHEMA = "transcribe-audio.plan0062-human-review-submission.v1"
MANIFEST_SCHEMA = "transcribe-audio.plan0062-human-review-manifest.v1"
RECEIPT_SCHEMA = "transcribe-audio.plan0062-human-review-receipt.v1"
SPEAKER_RE = re.compile(r"^SPEAKER_[1-9][0-9]*$")


class ContextualHumanReviewError(ValueError):
    """Raised when a review packet could weaken the private human gate."""


def _fail(message: str) -> None:
    raise ContextualHumanReviewError(message)


def _text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _person_label(person: Mapping[str, Any]) -> str:
    for key in ("display_name", "primary_name", "label", "name", "email"):
        value = _text(person.get(key))
        if value:
            return value
    return ""


def _suggestion_label(suggestion: Mapping[str, Any]) -> str:
    name = _text(suggestion.get("name"))
    email = _text(suggestion.get("email"))
    organization = _text(suggestion.get("organization"))
    primary = name or email or organization
    details = [value for value in (organization, email) if value and value != primary]
    return " | ".join([primary, *details])


def _option_token(prefix: str, value: Mapping[str, Any] | str) -> str:
    digest = canonical_artifact_hash(value)[:24]
    return f"{prefix}-{digest}"


def build_review_packet(
    p3_manifest: Mapping[str, Any],
    *,
    p3_content_sha256: str,
    identity_packets: Mapping[str, Mapping[str, Any]],
    acoustic_bundles: Mapping[str, AcousticEvidenceBundle],
    enrolled_subject_labels: Mapping[str, str],
) -> dict[str, Any]:
    """Create ten private review cards from the frozen P3 join."""

    if (
        p3_manifest.get("schema_version") != P3_MANIFEST_SCHEMA
        or p3_manifest.get("status") != "joined_pending_human_review"
        or canonical_artifact_hash(dict(p3_manifest)) != p3_content_sha256
        or int(p3_manifest.get("recording_count") or 0) != 3
        or int(p3_manifest.get("speaker_count") or 0) != 10
        or int(p3_manifest.get("evaluation_count") or 0) != 30
        or any((p3_manifest.get("negative_actions") or {}).values())
    ):
        _fail("The Plan 0062 P3 authority is invalid.")
    results = p3_manifest.get("results")
    if not isinstance(results, list) or tuple(
        str(item.get("document_id") or "") for item in results if isinstance(item, Mapping)
    ) != EXPECTED_DOCUMENTS:
        _fail("The Plan 0062 P3 cohort drifted.")
    if set(identity_packets) != set(EXPECTED_DOCUMENTS) or set(acoustic_bundles) != set(
        EXPECTED_DOCUMENTS
    ):
        _fail("Review inputs do not cover the exact Plan 0062 cohort.")

    cards: list[dict[str, Any]] = []
    for recording_ordinal, result in enumerate(results, start=1):
        document_id = str(result["document_id"])
        packet = identity_packets[document_id]
        if canonical_artifact_hash(dict(packet)) != result.get("identity_packet_sha256"):
            _fail("A private identity packet no longer matches P3.")
        people = {
            str(person.get("person_id") or ""): _person_label(person)
            for person in packet.get("people") or []
            if isinstance(person, Mapping) and str(person.get("person_id") or "")
        }
        join = result.get("join") if isinstance(result.get("join"), Mapping) else {}
        outcomes = join.get("review_outcomes")
        evaluations = join.get("evaluations")
        if not isinstance(outcomes, (list, tuple)) or not isinstance(
            evaluations, (list, tuple)
        ):
            _fail("A P3 joined result has no review outcomes.")
        evaluation_by_slot = {
            (str(item.get("speaker_ref") or ""), str(item.get("condition") or "")): item
            for item in evaluations
            if isinstance(item, Mapping)
        }
        acoustic = acoustic_bundles[document_id]
        acoustic_by_ref = {row.speaker_ref: row for row in acoustic.evidence}
        expected_speakers = EXPECTED_SPEAKER_COUNTS[document_id]
        if len(outcomes) != expected_speakers or len(acoustic_by_ref) != expected_speakers:
            _fail("A review case has an incomplete speaker denominator.")

        for outcome in outcomes:
            speaker_ref = str(outcome.get("speaker_ref") or "")
            if not SPEAKER_RE.fullmatch(speaker_ref) or speaker_ref not in acoustic_by_ref:
                _fail("A review card has an invalid acoustic speaker binding.")
            options: list[dict[str, str]] = []
            seen_labels: set[str] = set()

            acoustic_row = acoustic_by_ref[speaker_ref]
            acoustic_subject_id = str(acoustic_row.acoustic_subject_id or "")
            if acoustic_row.disposition != "abstain":
                acoustic_label = _text(enrolled_subject_labels.get(acoustic_subject_id))
                if not acoustic_subject_id or not acoustic_label:
                    _fail("A non-abstaining acoustic subject lacks reviewed display authority.")
                options.append(
                    {
                        "token": _option_token("enrolled", acoustic_subject_id),
                        "label": acoustic_label,
                        "source": "enrolled_voice_subject",
                    }
                )
                seen_labels.add(acoustic_label.casefold())

            context_person_id = str(outcome.get("context_person_id") or "")
            if context_person_id:
                label = _text(people.get(context_person_id))
                if not label:
                    _fail("A canonical context proposal lacks a private display label.")
                if label.casefold() not in seen_labels:
                    options.append(
                        {
                            "token": _option_token("canonical", context_person_id),
                            "label": label,
                            "source": "canonical_context_proposal",
                        }
                    )
                    seen_labels.add(label.casefold())

            suggestions = []
            for raw in outcome.get("suggestions") or []:
                if not isinstance(raw, Mapping):
                    continue
                label = _suggestion_label(raw)
                if not label:
                    continue
                suggestions.append(dict(raw))
                if label.casefold() in seen_labels:
                    continue
                options.append(
                    {
                        "token": _option_token("suggested", dict(raw)),
                        "label": label,
                        "source": "contextual_unlisted_suggestion",
                    }
                )
                seen_labels.add(label.casefold())

            condition_rows = []
            for condition in ("context_only", "acoustic_only", "combined"):
                evaluation = evaluation_by_slot.get((speaker_ref, condition))
                if not isinstance(evaluation, Mapping):
                    _fail("A review card lacks one of the three joined conditions.")
                condition_rows.append(
                    {
                        "condition": condition,
                        "outcome": str(evaluation.get("outcome") or ""),
                        "abstention_reason": str(
                            evaluation.get("abstention_reason") or ""
                        ),
                        "base_confidence": float(
                            evaluation.get("base_confidence") or 0.0
                        ),
                        "capped_confidence": float(
                            evaluation.get("capped_confidence") or 0.0
                        ),
                    }
                )
            slot_id = f"{document_id}::{speaker_ref}"
            cards.append(
                {
                    "slot_id": slot_id,
                    "document_id": document_id,
                    "recording_ordinal": recording_ordinal,
                    "speaker_ref": speaker_ref,
                    "source_speaker_label": str(
                        outcome.get("source_speaker_label") or ""
                    ),
                    "audio_path": f"media/recording-{recording_ordinal:02d}/{speaker_ref}.wav",
                    "context_status": str(outcome.get("context_status") or ""),
                    "context_reason": str(outcome.get("reason_code") or ""),
                    "suggestions": suggestions,
                    "acoustic": {
                        "disposition": acoustic_row.disposition,
                        "confidence_band": acoustic_row.confidence_band,
                        "score": float(acoustic_row.score),
                        "has_enrolled_subject": bool(acoustic_subject_id),
                    },
                    "conditions": condition_rows,
                    "options": options,
                }
            )
    if len(cards) != 10 or len({card["slot_id"] for card in cards}) != 10:
        _fail("The Plan 0062 review card denominator drifted.")
    packet = {
        "schema_version": PACKET_SCHEMA,
        "status": "awaiting_literal_human_review",
        "p3_content_sha256": p3_content_sha256,
        "recording_count": 3,
        "speaker_slot_count": 10,
        "preselected_decision_count": 0,
        "human_decision_count": 0,
        "cards": cards,
        "negative_actions": negative_action_vector(),
    }
    packet["content_sha256"] = canonical_artifact_hash(packet)
    return packet


def render_review_worksheet(packet: Mapping[str, Any]) -> str:
    """Render a self-contained client-only worksheet with relative WAV URLs."""

    cards = packet.get("cards")
    if (
        packet.get("schema_version") != PACKET_SCHEMA
        or packet.get("status") != "awaiting_literal_human_review"
        or canonical_artifact_hash({k: v for k, v in packet.items() if k != "content_sha256"})
        != packet.get("content_sha256")
        or not isinstance(cards, list)
        or len(cards) != 10
    ):
        _fail("The Plan 0062 review packet is invalid.")

    sections: list[str] = []
    for card in cards:
        slot = html.escape(str(card["slot_id"]), quote=True)
        option_rows = [dict(option) for option in card["options"]]
        options = "".join(
            f'<option value="{html.escape(str(option["token"]), quote=True)}">'
            f'{html.escape(str(option["label"]))} — {html.escape(str(option["source"]).replace("_", " "))}</option>'
            for option in option_rows
        )
        enrolled_options = [
            option
            for option in option_rows
            if option["source"] == "enrolled_voice_subject"
        ]
        contextual_options = [
            option
            for option in option_rows
            if option["source"]
            in {"canonical_context_proposal", "contextual_unlisted_suggestion"}
        ]
        contextual_tokens = {str(option["token"]) for option in contextual_options}
        for suggestion in card["suggestions"]:
            token = _option_token("suggested", dict(suggestion))
            if token not in contextual_tokens:
                contextual_options.append(
                    {
                        "token": token,
                        "label": _suggestion_label(suggestion),
                        "source": "contextual_unlisted_suggestion",
                    }
                )
                contextual_tokens.add(token)
        linked_options = "".join(
            f'<option value="linked:{html.escape(str(enrolled["token"]), quote=True)}:'
            f'{html.escape(str(contextual["token"]), quote=True)}">Same person — enrolled '
            f'{html.escape(str(enrolled["label"]))} + contextual '
            f'{html.escape(str(contextual["label"]))}</option>'
            for enrolled in enrolled_options
            for contextual in contextual_options
        )
        suggestions = "".join(
            f"<li>{html.escape(_suggestion_label(item))}</li>"
            for item in card["suggestions"]
        ) or "<li>No contextual name suggestion</li>"
        conditions = "".join(
            "<li><strong>"
            + html.escape(str(item["condition"]).replace("_", " ").title())
            + ":</strong> "
            + html.escape(str(item["outcome"]))
            + "; "
            + html.escape(str(item["abstention_reason"]) or "no abstention")
            + f'; confidence {float(item["capped_confidence"]):.2f}</li>'
            for item in card["conditions"]
        )
        acoustic = card["acoustic"]
        sections.append(
            f'''<article class="card" data-card data-slot="{slot}">
<h3>Recording {int(card["recording_ordinal"])} · {html.escape(str(card["speaker_ref"]))}</h3>
<p class="binding">Source diarization label: <code>{html.escape(str(card["source_speaker_label"]))}</code></p>
<audio controls preload="metadata"><source src="{html.escape(str(card["audio_path"]), quote=True)}" type="audio/wav"></audio>
<p><a href="{html.escape(str(card["audio_path"]), quote=True)}" target="_blank" rel="noopener">Open this WAV directly</a></p>
<p><strong>Existing contextual workflow:</strong> {html.escape(str(card["context_status"]))}; {html.escape(str(card["context_reason"]))}</p>
<ul>{suggestions}</ul>
<p><strong>Biometric shadow:</strong> {html.escape(str(acoustic["disposition"]))}; {html.escape(str(acoustic["confidence_band"]))}; score {float(acoustic["score"]):.2f}; enrolled subject {'present' if acoustic["has_enrolled_subject"] else 'absent'}.</p>
<details><summary>Three-condition readout</summary><ul>{conditions}</ul></details>
<label>Who is speaking?
<select data-decision data-slot="{slot}">
<option value="">Choose only after listening</option>{options}
{linked_options}
<option value="new_person">Different person — enter the name below</option>
<option value="unresolved">Unresolved / cannot determine</option>
</select></label>
<label>New/corrected person name <input data-new-name data-slot="{slot}" type="text" maxlength="160" placeholder="Enter for a new person or to correct a contextual name"></label>
<code>{slot}</code>
</article>'''
        )
    packet_json = json.dumps(
        {
            "submission_schema": SUBMISSION_SCHEMA,
            "p3_content_sha256": packet["p3_content_sha256"],
            "packet_content_sha256": packet["content_sha256"],
            "slots": [card["slot_id"] for card in cards],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).replace("</", "<\\/")
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="review-surface-schema" content="{WORKSHEET_SCHEMA}">
<title>Plan 0062 speaker identity review</title>
<style>
:root{{color-scheme:light dark;font-family:system-ui,sans-serif}}body{{max-width:1050px;margin:auto;padding:1rem}}.notice{{padding:1rem;border:2px solid #c77d00;border-radius:.7rem}}.card{{border:1px solid #888;border-radius:.7rem;padding:1rem;margin:1rem 0}}audio,select,input{{display:block;width:100%;max-width:48rem;margin:.45rem 0}}label{{display:block;margin:.8rem 0}}code{{overflow-wrap:anywhere}}button{{padding:.7rem 1rem;margin:.4rem .4rem .4rem 0}}textarea{{width:100%;min-height:18rem}}.error{{color:#b00020;font-weight:700}}
</style></head><body>
<h1>Context + biometric speaker review</h1>
<div class="notice"><strong>No decisions are preselected and nothing is applied.</strong> Listen to each clip. The name hints below come from the older calendar/transcript/provider workflow; the biometric line is independent evidence. Choosing a new person only records your review—it does not create a contact or voice profile.</div>
{''.join(sections)}
<h2>Copy review submission</h2><p id="error" class="error" role="alert"></p>
<button id="build" type="button">Build complete answer block</button><button id="copy" type="button">Copy answer block</button>
<textarea id="output" readonly></textarea>
<script id="review-config" type="application/json">{packet_json}</script>
<script>
const cfg=JSON.parse(document.getElementById('review-config').textContent);
function encodeName(value){{const bytes=new TextEncoder().encode(value);let binary='';for(const b of bytes)binary+=String.fromCharCode(b);return btoa(binary).replaceAll('+','-').replaceAll('/','_').replaceAll('=','');}}
function build(){{const rows=[];const error=document.getElementById('error');error.textContent='';for(const slot of cfg.slots){{const select=document.querySelector(`[data-decision][data-slot="${{CSS.escape(slot)}}"]`);const name=document.querySelector(`[data-new-name][data-slot="${{CSS.escape(slot)}}"]`).value.trim();let value=select.value;if(!value&&name){{value='new_person:'+encodeName(name);}}else if(!value){{error.textContent=`A decision is still blank: ${{slot}}`;return '';}}else if(value==='new_person'){{if(!name){{error.textContent=`Enter the new/corrected name for ${{slot}}`;return '';}}value='new_person:'+encodeName(name);}}else if(name&&(value.startsWith('suggested-')||value.startsWith('canonical-'))){{value='corrected:'+value+':'+encodeName(name);}}else if(name){{error.textContent=`Clear the name field or choose a contextual person to correct for ${{slot}}`;return '';}}rows.push(slot+'='+value);}}const header=[`PLAN0062_SCHEMA=${{cfg.submission_schema}}`,`PLAN0062_P3_CONTENT_SHA256=${{cfg.p3_content_sha256}}`,`PLAN0062_P4_CONTENT_SHA256=${{cfg.packet_content_sha256}}`];const text=[...header,...rows].join('\\n');document.getElementById('output').value=text;return text;}}
document.getElementById('build').addEventListener('click',build);document.getElementById('copy').addEventListener('click',async()=>{{const text=build();if(!text)return;try{{await navigator.clipboard.writeText(text)}}catch(_error){{const out=document.getElementById('output');out.focus();out.select();document.execCommand('copy');}}}});
</script></body></html>'''


def _write_private_text(path: Path, value: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(value)


def _copy_private_file(source: Path, target: Path) -> None:
    if source.is_symlink() or not source.is_file():
        _fail("A source review clip is unavailable.")
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with source.open("rb") as input_handle, os.fdopen(descriptor, "wb") as output_handle:
        shutil.copyfileobj(input_handle, output_handle)


def freeze_review_bundle(
    packet: Mapping[str, Any],
    *,
    audio_sources: Mapping[str, Path],
    runtime_root: Path,
) -> dict[str, Any]:
    """Freeze or replay the private worksheet directory and redacted receipt."""

    content_sha256 = str(packet.get("content_sha256") or "")
    root = runtime_root.expanduser().absolute()
    run = root / f"p4-human-review-{content_sha256[:24]}"
    preview = run / "preview"
    manifest_path = run / "private-manifest.json"
    receipt_path = run / "receipt.json"
    if receipt_path.exists():
        require_private_file(manifest_path, root)
        require_private_file(receipt_path, root)
        manifest = read_private_object(manifest_path)
        receipt = read_private_object(receipt_path)
        if (
            manifest.get("packet") != dict(packet)
            or receipt.get("manifest_sha256") != sha256_file(manifest_path)
            or receipt.get("worksheet_sha256") != sha256_file(preview / "review.html")
        ):
            _fail("The Plan 0062 P4 replay binding is invalid.")
        return {**receipt, "preview_path": str(preview), "idempotent_replay": True}
    if run.exists():
        _fail("The Plan 0062 P4 directory exists without a terminal receipt.")
    cards = packet.get("cards") or []
    expected_slots = {str(card["slot_id"]) for card in cards}
    if set(audio_sources) != expected_slots:
        _fail("The review clip set does not cover every exact speaker slot.")
    ensure_private_tree(root, run)
    ensure_private_tree(root, preview)
    audio_rows = []
    for card in cards:
        slot_id = str(card["slot_id"])
        target = preview / str(card["audio_path"])
        ensure_private_tree(root, target.parent)
        _copy_private_file(audio_sources[slot_id], target)
        audio_rows.append(
            {
                "slot_id": slot_id,
                "relative_path": str(card["audio_path"]),
                "bytes": target.stat().st_size,
                "sha256": sha256_file(target),
            }
        )
    worksheet = preview / "review.html"
    _write_private_text(worksheet, render_review_worksheet(packet))
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "status": "awaiting_literal_human_review",
        "packet": dict(packet),
        "audio_clips": audio_rows,
        "worksheet_sha256": sha256_file(worksheet),
        "preselected_decision_count": 0,
        "human_decision_count": 0,
        "apply_enabled": False,
        "negative_actions": negative_action_vector(),
    }
    write_immutable_private_json(manifest_path, manifest)
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "status": manifest["status"],
        "p3_content_sha256": packet["p3_content_sha256"],
        "content_sha256": content_sha256,
        "manifest_sha256": sha256_file(manifest_path),
        "worksheet_sha256": manifest["worksheet_sha256"],
        "recording_count": 3,
        "speaker_slot_count": 10,
        "audio_clip_count": len(audio_rows),
        "preselected_decision_count": 0,
        "human_decision_count": 0,
        "live_mutation_count": 0,
        "negative_actions_preserved": True,
    }
    write_immutable_private_json(receipt_path, receipt)
    return {**receipt, "preview_path": str(preview), "idempotent_replay": False}
