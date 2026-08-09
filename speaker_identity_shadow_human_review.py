from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import acoustic_plan0057
import speaker_identity_shadow_join_execution as plan0060
from acoustic_audio_derivatives import (
    canonical_artifact_hash,
    ensure_private_tree,
    read_private_object,
    require_private_file,
    sha256_file,
    write_immutable_private_json,
)


PLAN0060_ACTIVATION_SHA256 = (
    "08afc1b021a30f2a06f6e45bac88cec1b343def65b4e02261845ddff8667cf77"
)
PLAN0060_P4_CONTENT_SHA256 = (
    "6f6bb30f9073ad706c45561bbf56311457f53e714743d4d905469508ecb82320"
)
PLAN0060_P4_MANIFEST_SHA256 = (
    "e4883c01af517ee5db4387bdf01ddebd5d876158f7a05478a0968bab3e2808f4"
)
PLAN0060_TERMINAL_MANIFEST_SHA256 = (
    "f0eaac827ba19fc3b8bbd94dbe40b1efa4c525f5d351ba540238524767798a8d"
)
FROZEN_IDENTITY_STATE_SHA256 = (
    "64e0a7f44f59563ee848212a93d00e817be59c5471f035a96db7a75f8810924a"
)

WORKSHEET_SCHEMA = "transcribe-audio.plan0061-human-review-worksheet.v3"
WORKSHEET_MANIFEST_SCHEMA = (
    "transcribe-audio.plan0061-human-review-worksheet-manifest.v3"
)
WORKSHEET_RECEIPT_SCHEMA = (
    "transcribe-audio.plan0061-human-review-worksheet-receipt.v3"
)
DECISION_SUBMISSION_SCHEMA = (
    "transcribe-audio.plan0061-human-review-submission.v1"
)
MODULE_PATH = Path(__file__).name
DEFAULT_PLAN0060_ROOT = Path("~/.local/state/transcribe-audio/plan-0060")
DEFAULT_PLAN0060_ACOUSTIC_ROOT = Path(
    "~/.local/state/transcribe-audio/plan-0060/"
    "p2a-acoustic-08afc1b021a30f2a06f6e45b"
)
DEFAULT_RUNTIME_ROOT = Path("~/.local/state/transcribe-audio/plan-0061")
DEFAULT_LIVE_STORE_ROOT = Path("~/.transcripts")

SPEAKER_RE = re.compile(r"SPEAKER_[1-9][0-9]*")
OPAQUE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{2,255}")
SHA256_RE = re.compile(r"[a-f0-9]{64}")
CONDITIONS = ("context_only", "acoustic_only", "combined")
NON_PERSON_DECISIONS = ("not_listed", "unresolved")
EXPECTED_RECORDINGS = 3
EXPECTED_SPEAKERS = 10
EXPECTED_CONDITIONS = 30

NEGATIVE_ACTIONS = {
    "apply_speaker_assignments": False,
    "create_or_mutate_identities": False,
    "mutate_profiles_or_references": False,
    "write_live_knowledge": False,
    "write_external_provider": False,
    "write_graphiti": False,
    "restart_or_mutate_watchers": False,
    "enable_default_integration": False,
    "run_historical_reprocessing": False,
}


class Plan0061ReviewError(ValueError):
    """Raised when the human-review surface or submission is not exact."""


def _fail(message: str) -> None:
    raise Plan0061ReviewError(message)


def _git(arguments: Sequence[str], *, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", *arguments],
        cwd=Path(__file__).resolve().parent,
        capture_output=True,
        text=not binary,
        check=False,
    )
    if result.returncode:
        _fail("Repository authority is unavailable.")
    return result.stdout if binary else result.stdout.strip()


def _repository_authority() -> dict[str, Any]:
    if _git(["status", "--porcelain=v1", "--untracked-files=normal"]):
        _fail("Repository must be clean before private review preparation.")
    if str(
        _git(["rev-list", "--left-right", "--count", "@{upstream}...HEAD"])
    ).split() != ["0", "0"]:
        _fail("Repository must be upstream-even before review preparation.")
    commit = str(_git(["rev-parse", "HEAD"]))
    body = _git(["show", f"{commit}:{MODULE_PATH}"], binary=True)
    module_sha256 = sha256_file(Path(__file__).resolve())
    if not isinstance(body, bytes) or hashlib.sha256(body).hexdigest() != module_sha256:
        _fail("Committed Plan 0061 review authority drifted.")
    return {
        "commit": commit,
        "module_sha256": module_sha256,
        "clean": True,
        "upstream_ahead": 0,
        "upstream_behind": 0,
    }


def _text(value: Any, *, field: str, maximum: int = 500) -> str:
    normalized = " ".join(str(value or "").split())
    if not normalized or len(normalized) > maximum or any(
        character in normalized for character in "\r\n"
    ):
        _fail(f"{field} is invalid.")
    return normalized


def _opaque(value: Any, *, field: str) -> str:
    normalized = str(value or "")
    if not OPAQUE_RE.fullmatch(normalized):
        _fail(f"{field} is invalid.")
    return normalized


def _sha256(value: Any, *, field: str) -> str:
    normalized = str(value or "")
    if not SHA256_RE.fullmatch(normalized):
        _fail(f"{field} is invalid.")
    return normalized


def _string_list(value: Any, *, field: str) -> list[str]:
    if not isinstance(value, list):
        _fail(f"{field} must be a list.")
    return [_text(item, field=field, maximum=1000) for item in value]


def _candidate_label(person_id: str, labels: Mapping[str, str]) -> str:
    return labels.get(person_id, "Unavailable candidate")


def _condition_view(raw: Mapping[str, Any], labels: Mapping[str, str]) -> dict[str, Any]:
    condition = str(raw.get("condition") or "")
    if condition not in CONDITIONS:
        _fail("A condition view is invalid.")
    outcome = _text(raw.get("outcome"), field="condition outcome", maximum=40)
    proposed = raw.get("proposed_person_id")
    if proposed is not None:
        proposed = _opaque(proposed, field="proposed person ID")
    alternatives = raw.get("alternative_person_ids")
    if not isinstance(alternatives, list):
        _fail("Condition alternatives must be a list.")
    alternative_ids = [
        _opaque(item, field="alternative person ID") for item in alternatives
    ]
    factors = raw.get("factors")
    if not isinstance(factors, list):
        _fail("Condition factors must be a list.")
    normalized_factors: list[dict[str, Any]] = []
    for factor in factors:
        if not isinstance(factor, Mapping):
            _fail("A condition factor is invalid.")
        evidence_ids = factor.get("evidence_ids")
        independence_groups = factor.get("independence_groups")
        if not isinstance(evidence_ids, list) or not isinstance(independence_groups, list):
            _fail("A condition factor has invalid evidence bindings.")
        score = factor.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            _fail("A condition factor score is invalid.")
        normalized_factors.append(
            {
                "factor_type": _text(
                    factor.get("factor_type"), field="factor type", maximum=40
                ),
                "score": round(float(score), 4),
                "evidence_count": len(evidence_ids),
                "independence_group_count": len(independence_groups),
            }
        )
    base_confidence = raw.get("base_confidence")
    capped_confidence = raw.get("capped_confidence")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in (base_confidence, capped_confidence)
    ):
        _fail("Condition confidence is invalid.")
    source_failures = raw.get("source_failures")
    if not isinstance(source_failures, list):
        _fail("Condition source failures must be a list.")
    normalized_failures = [
        " / ".join(_text(part, field="source failure", maximum=300) for part in item)
        if isinstance(item, (list, tuple))
        else _text(item, field="source failure", maximum=600)
        for item in source_failures
    ]
    cap_reasons = _string_list(
        raw.get("confidence_cap_reasons"), field="confidence cap reason"
    )
    abstention = raw.get("abstention_reason")
    abstention_reason = (
        _text(abstention, field="abstention reason", maximum=300)
        if abstention is not None
        else "None"
    )
    contradiction_ids = raw.get("contradiction_evidence_ids") or []
    if not isinstance(contradiction_ids, list):
        _fail("Condition contradiction evidence must be a list.")
    return {
        "condition": condition,
        "outcome": outcome,
        "proposed_label": (
            _candidate_label(proposed, labels) if proposed is not None else "None"
        ),
        "alternative_labels": [
            _candidate_label(person_id, labels) for person_id in alternative_ids
        ],
        "base_confidence": round(float(base_confidence), 4),
        "capped_confidence": round(float(capped_confidence), 4),
        "confidence_cap_reasons": cap_reasons,
        "abstention_reason": abstention_reason,
        "source_failures": normalized_failures,
        "contradiction_count": len(contradiction_ids),
        "factors": normalized_factors,
    }


def normalized_review_cases(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Validate the exact P4 denominator and return minimum-copy review cases."""

    if (
        manifest.get("schema_version") != plan0060.P4_MANIFEST_VERSION
        or manifest.get("activation_sha256") != PLAN0060_ACTIVATION_SHA256
        or manifest.get("recording_count") != EXPECTED_RECORDINGS
        or manifest.get("speaker_slot_count") != EXPECTED_SPEAKERS
        or manifest.get("condition_count") != EXPECTED_CONDITIONS
        or manifest.get("human_decision_count") != 0
        or manifest.get("preselected_decision_count") != 0
        or manifest.get("apply_enabled") is not False
        or manifest.get("human_gold_read") is not False
    ):
        _fail("The Plan 0060 P4 review authority drifted.")
    negative_actions = manifest.get("negative_actions")
    if not isinstance(negative_actions, Mapping) or not negative_actions or any(
        value is not False for value in negative_actions.values()
    ):
        _fail("The P4 negative-action boundary drifted.")
    raw_cases = manifest.get("cases")
    if not isinstance(raw_cases, list) or len(raw_cases) != EXPECTED_RECORDINGS:
        _fail("The P4 recording denominator drifted.")

    normalized: list[dict[str, Any]] = []
    seen_documents: set[str] = set()
    seen_recordings: set[str] = set()
    seen_slots: set[str] = set()
    seen_evaluations: set[str] = set()
    condition_count = 0
    for case_ordinal, raw_case in enumerate(raw_cases, start=1):
        if not isinstance(raw_case, Mapping):
            _fail("A P4 review case is invalid.")
        document_id = _opaque(raw_case.get("document_id"), field="document ID")
        recording_id = _opaque(raw_case.get("recording_id"), field="recording ID")
        if document_id in seen_documents or recording_id in seen_recordings:
            _fail("The P4 case set contains a duplicate recording or document.")
        seen_documents.add(document_id)
        seen_recordings.add(recording_id)

        raw_candidates = raw_case.get("candidate_options")
        if not isinstance(raw_candidates, list) or not raw_candidates:
            _fail("A P4 case has no candidate options.")
        candidates: list[dict[str, str]] = []
        labels: dict[str, str] = {}
        for raw_candidate in raw_candidates:
            if not isinstance(raw_candidate, Mapping):
                _fail("A P4 candidate option is invalid.")
            person_id = _opaque(raw_candidate.get("person_id"), field="candidate person ID")
            label = _text(raw_candidate.get("label"), field="candidate label", maximum=160)
            if person_id in labels:
                _fail("A P4 candidate is duplicated within one recording.")
            labels[person_id] = label
            candidates.append({"person_id": person_id, "label": label})

        raw_slots = raw_case.get("speaker_slots")
        if not isinstance(raw_slots, list) or not raw_slots:
            _fail("A P4 case has no speaker slots.")
        slots: list[dict[str, Any]] = []
        expected_allowed = {*labels, *NON_PERSON_DECISIONS}
        for raw_slot in raw_slots:
            if not isinstance(raw_slot, Mapping):
                _fail("A P4 speaker slot is invalid.")
            speaker_ref = str(raw_slot.get("speaker_ref") or "")
            if not SPEAKER_RE.fullmatch(speaker_ref):
                _fail("A P4 speaker reference is invalid.")
            slot_id = f"{document_id}::{speaker_ref}"
            if slot_id in seen_slots:
                _fail("The P4 packet contains a duplicate decision slot.")
            seen_slots.add(slot_id)
            allowed = raw_slot.get("allowed_decisions")
            if (
                not isinstance(allowed, list)
                or len(allowed) != len(set(str(item) for item in allowed))
                or set(str(item) for item in allowed) != expected_allowed
                or raw_slot.get("selected_person_id") is not None
            ):
                _fail("A P4 decision slot is preselected or has drifted choices.")

            raw_conditions = raw_slot.get("conditions")
            if not isinstance(raw_conditions, list) or len(raw_conditions) != 3:
                _fail("A P4 speaker slot must contain three condition views.")
            conditions = []
            condition_names = set()
            for raw_condition in raw_conditions:
                if not isinstance(raw_condition, Mapping):
                    _fail("A P4 condition view is invalid.")
                evaluation_id = _opaque(
                    raw_condition.get("evaluation_id"), field="evaluation ID"
                )
                if evaluation_id in seen_evaluations:
                    _fail("The P4 packet contains a duplicate evaluation.")
                seen_evaluations.add(evaluation_id)
                condition = _condition_view(raw_condition, labels)
                condition_names.add(condition["condition"])
                conditions.append(condition)
            if condition_names != set(CONDITIONS):
                _fail("A P4 speaker slot has a missing or duplicate condition.")
            condition_count += len(conditions)

            raw_acoustic = raw_slot.get("acoustic")
            if not isinstance(raw_acoustic, Mapping):
                _fail("A P4 acoustic summary is invalid.")
            score = raw_acoustic.get("score")
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                _fail("A P4 acoustic score is invalid.")
            slots.append(
                {
                    "slot_id": slot_id,
                    "speaker_ref": speaker_ref,
                    "audio_url": (
                        f"media/recording-{case_ordinal:02d}/{speaker_ref}.wav"
                    ),
                    "allowed_decisions": [candidate["person_id"] for candidate in candidates]
                    + list(NON_PERSON_DECISIONS),
                    "acoustic": {
                        "disposition": _text(
                            raw_acoustic.get("disposition"),
                            field="acoustic disposition",
                            maximum=80,
                        ),
                        "confidence_band": _text(
                            raw_acoustic.get("confidence_band"),
                            field="acoustic confidence band",
                            maximum=40,
                        ),
                        "score": round(float(score), 4),
                        "supporting_unit_count": int(
                            raw_acoustic.get("supporting_unit_count") or 0
                        ),
                        "opposing_unit_count": int(
                            raw_acoustic.get("opposing_unit_count") or 0
                        ),
                        "insufficient_unit_count": int(
                            raw_acoustic.get("insufficient_unit_count") or 0
                        ),
                    },
                    "conditions": conditions,
                }
            )

        warnings = raw_case.get("warnings")
        source_failures = raw_case.get("source_failures")
        scopes = raw_case.get("scopes")
        if not isinstance(warnings, list) or not isinstance(source_failures, list):
            _fail("P4 case warnings or source failures are invalid.")
        if not isinstance(scopes, list):
            _fail("P4 case scopes are invalid.")
        scope_summaries = []
        for raw_scope in scopes:
            if not isinstance(raw_scope, Mapping):
                _fail("A P4 provider scope is invalid.")
            capabilities = raw_scope.get("capabilities")
            if not isinstance(capabilities, list):
                _fail("A P4 provider scope has invalid capabilities.")
            scope_summaries.append(
                {
                    "source_type": _text(
                        raw_scope.get("source_type"), field="scope source type", maximum=80
                    ),
                    "capabilities": [
                        _text(item, field="scope capability", maximum=80)
                        for item in capabilities
                    ],
                    "max_provider_calls": int(raw_scope.get("max_provider_calls") or 0),
                    "max_records": int(raw_scope.get("max_records") or 0),
                }
            )
        normalized.append(
            {
                "case_ordinal": case_ordinal,
                "document_id": document_id,
                "recording_id": recording_id,
                "candidates": candidates,
                "warnings": [
                    _text(item, field="case warning", maximum=600) for item in warnings
                ],
                "source_failures": [
                    _text(item, field="case source failure", maximum=600)
                    for item in source_failures
                ],
                "scopes": scope_summaries,
                "slots": slots,
            }
        )
    if (
        len(seen_slots) != EXPECTED_SPEAKERS
        or len(seen_evaluations) != EXPECTED_CONDITIONS
        or condition_count != EXPECTED_CONDITIONS
    ):
        _fail("The P4 review denominator is incomplete.")
    return normalized


def _json_for_script(value: Any) -> str:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _condition_html(condition: Mapping[str, Any]) -> str:
    factor_rows = "".join(
        "<li>"
        f"{html.escape(str(factor['factor_type']))}: score {factor['score']:.4f}; "
        f"{factor['evidence_count']} evidence item(s); "
        f"{factor['independence_group_count']} independence group(s)"
        "</li>"
        for factor in condition["factors"]
    ) or "<li>No evidence factors.</li>"
    alternatives = ", ".join(condition["alternative_labels"]) or "None"
    cap_reasons = ", ".join(condition["confidence_cap_reasons"]) or "None"
    failures = ", ".join(condition["source_failures"]) or "None"
    label = str(condition["condition"]).replace("_", " ").title()
    return (
        '<section class="condition">'
        f"<h4>{html.escape(label)}</h4>"
        f"<p><strong>Outcome:</strong> {html.escape(str(condition['outcome']))}; "
        f"<strong>proposal:</strong> {html.escape(str(condition['proposed_label']))}</p>"
        f"<p><strong>Alternatives:</strong> {html.escape(alternatives)}</p>"
        f"<p><strong>Confidence:</strong> {condition['base_confidence']:.4f} base / "
        f"{condition['capped_confidence']:.4f} capped</p>"
        f"<p><strong>Cap reasons:</strong> {html.escape(cap_reasons)}</p>"
        f"<p><strong>Abstention reason:</strong> "
        f"{html.escape(str(condition['abstention_reason']))}</p>"
        f"<p><strong>Contradiction evidence:</strong> "
        f"{condition['contradiction_count']} item(s)</p>"
        f"<p><strong>Source failures:</strong> {html.escape(failures)}</p>"
        f"<ul>{factor_rows}</ul></section>"
    )


def render_review_worksheet(manifest: Mapping[str, Any]) -> str:
    """Render the exact client-only Plan 0061 human-review worksheet."""

    cases = normalized_review_cases(manifest)
    sections: list[str] = []
    slot_order: list[str] = []
    allowed_by_slot: dict[str, list[str]] = {}
    for case in cases:
        candidate_options = "".join(
            f'<option value="{html.escape(candidate["person_id"], quote=True)}">'
            f'{html.escape(candidate["label"])}</option>'
            for candidate in case["candidates"]
        )
        scope_rows = "".join(
            "<li>"
            f"{html.escape(scope['source_type'])}: "
            f"{html.escape(', '.join(scope['capabilities']) or 'no capabilities')}; "
            f"budget {scope['max_provider_calls']} call(s), {scope['max_records']} record(s)"
            "</li>"
            for scope in case["scopes"]
        ) or "<li>No provider scope rows.</li>"
        warnings = ", ".join(case["warnings"]) or "None"
        failures = ", ".join(case["source_failures"]) or "None"
        slot_sections = []
        for slot in case["slots"]:
            slot_order.append(slot["slot_id"])
            allowed_by_slot[slot["slot_id"]] = list(slot["allowed_decisions"])
            acoustic = slot["acoustic"]
            condition_sections = "".join(
                _condition_html(condition) for condition in slot["conditions"]
            )
            audio_url = html.escape(slot["audio_url"], quote=True)
            slot_sections.append(
                '<article class="slot" data-review-slot '
                f'data-slot-id="{html.escape(slot["slot_id"], quote=True)}">'
                f'<h3>{html.escape(slot["speaker_ref"])}</h3>'
                '<p class="listen"><strong>Listen to this speaker clip:</strong></p>'
                '<audio controls preload="metadata" data-review-audio '
                f'data-slot-id="{html.escape(slot["slot_id"], quote=True)}">'
                f'<source src="{audio_url}" type="audio/wav"></audio>'
                f'<p><a class="audio-fallback" href="{audio_url}" '
                'target="_blank" rel="noopener">Open this speaker WAV directly</a></p>'
                "<p><strong>Acoustic summary:</strong> "
                f"{html.escape(acoustic['disposition'])}; "
                f"{html.escape(acoustic['confidence_band'])} confidence; "
                f"score {acoustic['score']:.4f}; "
                f"{acoustic['supporting_unit_count']} supporting, "
                f"{acoustic['opposing_unit_count']} opposing, "
                f"{acoustic['insufficient_unit_count']} insufficient unit(s).</p>"
                '<div class="conditions">'
                f"{condition_sections}</div>"
                f'<label for="decision-{len(slot_order)}">Canonical identity decision</label>'
                f'<select id="decision-{len(slot_order)}" data-decision '
                f'data-slot-id="{html.escape(slot["slot_id"], quote=True)}">'
                '<option value="">Select only after listening and review</option>'
                f"{candidate_options}"
                '<option value="not_listed">Person is not listed</option>'
                '<option value="unresolved">Unresolved / cannot determine</option>'
                "</select>"
                f'<code>{html.escape(slot["slot_id"])}</code>'
                "</article>"
            )
        sections.append(
            '<section class="case">'
            f"<h2>Recording {case['case_ordinal']}</h2>"
            "<p>Listen to each bound speaker clip directly below, compare all three frozen "
            "evidence conditions, then make one canonical decision per speaker.</p>"
            f"<p><strong>Case warnings:</strong> {html.escape(warnings)}</p>"
            f"<p><strong>Case source failures:</strong> {html.escape(failures)}</p>"
            f"<details><summary>Bounded evidence scopes</summary><ul>{scope_rows}</ul></details>"
            f"{''.join(slot_sections)}</section>"
        )

    headers = [
        f"PLAN0061_SCHEMA={DECISION_SUBMISSION_SCHEMA}",
        f"PLAN0061_P4_CONTENT_SHA256={PLAN0060_P4_CONTENT_SHA256}",
        f"PLAN0061_P4_MANIFEST_SHA256={PLAN0060_P4_MANIFEST_SHA256}",
    ]
    script = f"""
const slotOrder = {_json_for_script(slot_order)};
const allowedBySlot = {_json_for_script(allowed_by_slot)};
const headers = {_json_for_script(headers)};
const progress = document.querySelector('#review-progress');
const exportButton = document.querySelector('#prepare-export');
const copyButton = document.querySelector('#copy-export');
const downloadLink = document.querySelector('#download-export');
const answerBlock = document.querySelector('#answer-block');
const exportStatus = document.querySelector('#export-status');

function selections() {{
  const values = new Map();
  for (const control of document.querySelectorAll('[data-decision]')) {{
    const slotId = control.dataset.slotId;
    const value = control.value;
    if (value && allowedBySlot[slotId]?.includes(value)) values.set(slotId, value);
  }}
  return values;
}}

function updateProgress() {{
  const complete = selections().size;
  progress.textContent = `${{complete}} / ${{slotOrder.length}} decisions complete`;
  exportButton.disabled = complete !== slotOrder.length;
  if (complete !== slotOrder.length) {{
    answerBlock.value = '';
    copyButton.disabled = true;
    downloadLink.hidden = true;
  }}
}}

function prepareExport() {{
  const values = selections();
  if (values.size !== slotOrder.length) {{
    exportStatus.textContent = `${{slotOrder.length - values.size}} decision(s) remain.`;
    return;
  }}
  const rows = [...headers];
  for (const slotId of slotOrder) {{
    const value = values.get(slotId);
    if (!value || !allowedBySlot[slotId]?.includes(value)) {{
      exportStatus.textContent = `Invalid decision for ${{slotId}}.`;
      return;
    }}
    rows.push(`${{slotId}}=${{value}}`);
  }}
  answerBlock.value = rows.join('\\n');
  copyButton.disabled = false;
  const blob = new Blob([answerBlock.value + '\\n'], {{type: 'text/plain'}});
  if (downloadLink.dataset.objectUrl) URL.revokeObjectURL(downloadLink.dataset.objectUrl);
  downloadLink.dataset.objectUrl = URL.createObjectURL(blob);
  downloadLink.href = downloadLink.dataset.objectUrl;
  downloadLink.hidden = false;
  exportStatus.textContent = `Prepared ${{slotOrder.length}} exact decisions. Nothing was submitted or applied.`;
}}

for (const control of document.querySelectorAll('[data-decision]')) {{
  control.addEventListener('change', updateProgress);
}}
exportButton.addEventListener('click', prepareExport);
copyButton.addEventListener('click', async () => {{
  if (!answerBlock.value) return;
  try {{
    await navigator.clipboard.writeText(answerBlock.value);
    exportStatus.textContent = 'Copied the complete decision block. Paste it into the preview feedback or chat.';
  }} catch (error) {{
    answerBlock.focus();
    answerBlock.select();
    exportStatus.textContent = 'Clipboard unavailable. The complete block is selected for manual copy.';
  }}
}});
document.querySelector('#clear-review').addEventListener('click', () => {{
  for (const control of document.querySelectorAll('[data-decision]')) control.value = '';
  answerBlock.value = '';
  copyButton.disabled = true;
  downloadLink.hidden = true;
  exportStatus.textContent = 'Choices cleared locally. Nothing was submitted or applied.';
  updateProgress();
}});
updateProgress();
"""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="review-surface-schema" content="{WORKSHEET_SCHEMA}">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; media-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>Plan 0061 joined speaker-identity review</title><style>
:root{{--ink:#18202a;--muted:#536273;--line:#c9d3df;--paper:#fff;--wash:#f2f5f8;--accent:#315c9c;--warn:#8a5a00}}
body{{font:16px/1.45 system-ui,sans-serif;max-width:1180px;margin:2rem auto;padding:0 1rem;background:var(--wash);color:var(--ink)}}
h1,h2,h3,h4{{line-height:1.2}}.notice,.case,.slot,.export{{background:var(--paper);border:1px solid var(--line);border-radius:12px;padding:1rem;margin:1rem 0}}
.notice{{border-left:6px solid var(--warn)}}.case{{border-top:6px solid var(--accent)}}.slot{{margin-left:.25rem}}.conditions{{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:.8rem}}
.condition{{background:#f8fafc;border:1px solid var(--line);border-radius:9px;padding:.8rem}}.condition p,.condition li{{font-size:.92rem}}label{{display:block;font-weight:700;margin-top:1rem}}
select,textarea,button,.download{{font:inherit}}select,textarea{{box-sizing:border-box;width:100%;padding:.7rem;margin:.4rem 0;border:1px solid #8797a8;border-radius:7px}}
textarea{{min-height:18rem}}button,.download{{display:inline-block;padding:.7rem 1rem;margin:.35rem .35rem .35rem 0}}button:disabled{{opacity:.5}}code{{display:block;word-break:break-all;color:var(--muted);margin-top:.5rem}}
.listen{{margin-bottom:.35rem}}audio{{display:block;width:100%;max-width:44rem;margin:.35rem 0}}.audio-fallback{{font-size:.92rem}}
.sticky{{position:sticky;top:0;z-index:2;background:#eef3f8;border:1px solid var(--line);border-radius:9px;padding:.7rem 1rem}}:focus-visible{{outline:3px solid #1769d2;outline-offset:2px}}
</style></head><body><main>
<h1>Plan 0061 joined speaker-identity review</h1>
<div class="notice"><strong>Human gold, not an apply screen.</strong> Listen to the bound speaker clip and review all three frozen evidence conditions before choosing. This page stores choices only in this browser tab. It cannot submit decisions, apply assignments, create identities, update profiles or references, write providers or Graphiti, restart watchers, or reprocess history.</div>
<div class="sticky"><strong id="review-progress" aria-live="polite">0 / {EXPECTED_SPEAKERS} decisions complete</strong></div>
{''.join(sections)}
<section class="export"><h2>Export the complete decision block</h2>
<p>Export stays disabled until all {EXPECTED_SPEAKERS} slots have a literal choice. Copy the resulting block into this preview's feedback or the chat. Approval alone does not count as identity gold.</p>
<button id="prepare-export" type="button" disabled>Prepare exact decision block</button>
<button id="copy-export" type="button" disabled>Copy decision block</button>
<a id="download-export" class="download" download="plan0061-human-decisions.txt" hidden>Download decision block</a>
<button id="clear-review" type="button">Clear local choices</button>
<output id="export-status" aria-live="assertive">Nothing has been submitted or applied.</output>
<label for="answer-block">Complete hash-bound decision block</label>
<textarea id="answer-block" readonly></textarea></section>
</main><script>{script}</script></body></html>"""


def parse_decision_block(
    answer_text: str, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Parse 10/10 exact decisions without accepting labels or partial gold."""

    cases = normalized_review_cases(manifest)
    expected: list[str] = []
    allowed: dict[str, set[str]] = {}
    for case in cases:
        for slot in case["slots"]:
            expected.append(slot["slot_id"])
            allowed[slot["slot_id"]] = set(slot["allowed_decisions"])
    lines = [line.strip() for line in str(answer_text or "").splitlines() if line.strip()]
    headers = [
        f"PLAN0061_SCHEMA={DECISION_SUBMISSION_SCHEMA}",
        f"PLAN0061_P4_CONTENT_SHA256={PLAN0060_P4_CONTENT_SHA256}",
        f"PLAN0061_P4_MANIFEST_SHA256={PLAN0060_P4_MANIFEST_SHA256}",
    ]
    if lines[:3] != headers or len(lines) != len(headers) + len(expected):
        _fail("The decision block is stale, incomplete, or has invalid headers.")
    decisions: dict[str, str] = {}
    for line in lines[3:]:
        if "=" not in line:
            _fail("A decision row is malformed.")
        slot_id, value = line.rsplit("=", 1)
        if slot_id in decisions or slot_id not in allowed or value not in allowed[slot_id]:
            _fail("A decision row is duplicate, unknown, or out of set.")
        decisions[slot_id] = value
    if list(decisions) != expected or set(decisions) != set(expected):
        _fail("The decision block does not preserve the exact slot order and denominator.")
    core = {
        "schema_version": DECISION_SUBMISSION_SCHEMA,
        "status": "complete_operator_decisions_preview",
        "plan0060_activation_sha256": PLAN0060_ACTIVATION_SHA256,
        "p4_content_sha256": PLAN0060_P4_CONTENT_SHA256,
        "p4_manifest_sha256": PLAN0060_P4_MANIFEST_SHA256,
        "decision_count": len(decisions),
        "decisions": [
            {"slot_id": slot_id, "decision": decisions[slot_id]}
            for slot_id in expected
        ],
        "applied_assignments": False,
        "created_or_mutated_identities": False,
        "mutated_profiles_or_references": False,
        "wrote_live_knowledge": False,
        "wrote_external_provider": False,
        "wrote_graphiti": False,
    }
    return {**core, "content_sha256": canonical_artifact_hash(core)}


def _live_baseline(live_store_root: Path) -> dict[str, Any]:
    database = live_store_root.expanduser().absolute() / "transcripts.sqlite3"
    if not database.is_file() or database.is_symlink():
        _fail("The live transcript database is unavailable.")
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        quick_check = str(connection.execute("PRAGMA quick_check").fetchone()[0])
        counts = {
            "documents": int(connection.execute("SELECT count(*) FROM documents").fetchone()[0]),
            "contacts": int(connection.execute("SELECT count(*) FROM contacts").fetchone()[0]),
            "speaker_assignments": int(
                connection.execute("SELECT count(*) FROM speaker_assignments").fetchone()[0]
            ),
            "knowledge_tables": int(
                connection.execute(
                    "SELECT count(*) FROM sqlite_master WHERE type='table' AND name LIKE 'knowledge_%'"
                ).fetchone()[0]
            ),
        }
    if quick_check != "ok" or counts != {
        "documents": 466,
        "contacts": 2,
        "speaker_assignments": 3,
        "knowledge_tables": 0,
    }:
        _fail("The frozen live database baseline drifted.")
    identity_state = acoustic_plan0057._current_identity_state()
    if identity_state.get("snapshot_sha256") != FROZEN_IDENTITY_STATE_SHA256:
        _fail("The frozen identity/profile/reference state drifted.")
    services: dict[str, dict[str, Any]] = {}
    for service in ("transcribe-watch.service", "transcripts.service"):
        result = subprocess.run(
            [
                "systemctl",
                "--user",
                "show",
                service,
                "--property=ActiveState",
                "--property=SubState",
                "--property=NRestarts",
                "--no-pager",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        values = dict(
            line.split("=", 1)
            for line in result.stdout.splitlines()
            if "=" in line
        )
        if result.returncode or values != {
            "ActiveState": "active",
            "SubState": "running",
            "NRestarts": "0",
        }:
            _fail(f"The frozen {service} service baseline drifted.")
        services[service] = {
            "active_state": "active",
            "sub_state": "running",
            "restart_count": 0,
        }
    return {
        "quick_check": quick_check,
        **counts,
        "identity_state_sha256": FROZEN_IDENTITY_STATE_SHA256,
        "services": services,
    }


def _validated_live_source(
    *, plan0060_root: Path, live_store_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = plan0060_root.expanduser().absolute()
    receipt = plan0060.replay_review_packet(
        runtime_root=root,
        activation_sha256=PLAN0060_ACTIVATION_SHA256,
    )
    manifest_path = Path(str(receipt.get("manifest_path") or ""))
    require_private_file(manifest_path, root)
    if (
        receipt.get("content_sha256") != PLAN0060_P4_CONTENT_SHA256
        or receipt.get("manifest_sha256") != PLAN0060_P4_MANIFEST_SHA256
        or sha256_file(manifest_path) != PLAN0060_P4_MANIFEST_SHA256
    ):
        _fail("The sealed Plan 0060 P4 receipt drifted.")
    manifest = read_private_object(manifest_path)
    normalized_review_cases(manifest)
    terminal_path = (
        root
        / f"terminal-review-ready-{PLAN0060_ACTIVATION_SHA256[:24]}"
        / "private-manifest.json"
    )
    require_private_file(terminal_path, root)
    if sha256_file(terminal_path) != PLAN0060_TERMINAL_MANIFEST_SHA256:
        _fail("The Plan 0060 terminal authority drifted.")
    return manifest, {
        "plan0060_activation_sha256": PLAN0060_ACTIVATION_SHA256,
        "p4_content_sha256": PLAN0060_P4_CONTENT_SHA256,
        "p4_manifest_sha256": PLAN0060_P4_MANIFEST_SHA256,
        "terminal_manifest_sha256": PLAN0060_TERMINAL_MANIFEST_SHA256,
        "live": _live_baseline(live_store_root),
    }


def _worksheet_paths(runtime_root: Path, worksheet_sha256: str) -> dict[str, Path]:
    root = runtime_root.expanduser().absolute()
    run = root / f"review-worksheet-{worksheet_sha256[:24]}"
    bundle = run / "preview"
    return {
        "root": root,
        "run": run,
        "bundle": bundle,
        "worksheet": bundle / "review.html",
        "media": bundle / "media",
        "manifest": run / "private-manifest.json",
        "receipt": run / "receipt.json",
    }


def _validated_audio_sources(
    manifest: Mapping[str, Any], acoustic_root: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Bind each review slot to one frozen Plan 0060 P2A speaker clip."""

    cases = normalized_review_cases(manifest)
    root = acoustic_root.expanduser().absolute()
    authority_path = root / "private-manifest.json"
    require_private_file(authority_path, root)
    authority = read_private_object(authority_path)
    negative_actions = authority.get("negative_actions")
    results = authority.get("results")
    if (
        authority.get("schema_version")
        != "transcribe-audio.plan0060-p2a-acoustic-manifest.v1"
        or authority.get("status") != "acoustic_lane_complete"
        or authority.get("activation_sha256") != PLAN0060_ACTIVATION_SHA256
        or authority.get("recording_count") != EXPECTED_RECORDINGS
        or authority.get("speaker_ref_count") != EXPECTED_SPEAKERS
        or not isinstance(negative_actions, Mapping)
        or not negative_actions
        or any(value is not False for value in negative_actions.values())
        or not isinstance(results, list)
        or len(results) != EXPECTED_RECORDINGS
    ):
        _fail("The frozen Plan 0060 acoustic clip authority drifted.")
    result_by_document = {
        str(result.get("document_id") or ""): result
        for result in results
        if isinstance(result, Mapping)
    }
    if len(result_by_document) != EXPECTED_RECORDINGS:
        _fail("The acoustic authority document set drifted.")

    proposal_paths = sorted((root / "sources").glob("*/acoustic/proposals.json"))
    if len(proposal_paths) != EXPECTED_RECORDINGS:
        _fail("The acoustic proposal source set drifted.")
    source_by_document: dict[str, Path] = {}
    speaker_refs_by_document: dict[str, set[str]] = {}
    for proposal_path in proposal_paths:
        require_private_file(proposal_path, root)
        proposal = read_private_object(proposal_path)
        document_id = _opaque(proposal.get("document_id"), field="acoustic document ID")
        rows = proposal.get("rows")
        if (
            document_id in source_by_document
            or not isinstance(rows, list)
            or proposal.get("speaker_count") != len(rows)
        ):
            _fail("A frozen acoustic proposal set drifted.")
        speaker_refs = {
            str(row.get("speaker_ref") or "")
            for row in rows
            if isinstance(row, Mapping)
        }
        if len(speaker_refs) != len(rows) or any(
            not SPEAKER_RE.fullmatch(item) for item in speaker_refs
        ):
            _fail("A frozen acoustic proposal speaker set drifted.")
        source_by_document[document_id] = proposal_path.parent.parent
        speaker_refs_by_document[document_id] = speaker_refs

    clips: list[dict[str, Any]] = []
    for case in cases:
        document_id = case["document_id"]
        result = result_by_document.get(document_id)
        source_dir = source_by_document.get(document_id)
        expected_refs = {slot["speaker_ref"] for slot in case["slots"]}
        if (
            not isinstance(result, Mapping)
            or source_dir is None
            or result.get("recording_id") != case["recording_id"]
            or result.get("speaker_ref_count") != len(expected_refs)
            or speaker_refs_by_document.get(document_id) != expected_refs
        ):
            _fail("The acoustic clips do not match the sealed review cases.")
        for slot in case["slots"]:
            source_path = source_dir / "acoustic" / "clips" / f"{slot['speaker_ref']}.wav"
            require_private_file(source_path, root)
            source_sha256 = sha256_file(source_path)
            byte_count = source_path.stat().st_size
            if byte_count <= 44:
                _fail("A frozen acoustic speaker clip is empty.")
            clips.append(
                {
                    "slot_id": slot["slot_id"],
                    "document_id": document_id,
                    "speaker_ref": slot["speaker_ref"],
                    "relative_path": slot["audio_url"],
                    "source_path": source_path,
                    "sha256": source_sha256,
                    "bytes": byte_count,
                }
            )
    if len(clips) != EXPECTED_SPEAKERS:
        _fail("The acoustic speaker clip denominator drifted.")
    return clips, {
        "schema_version": authority["schema_version"],
        "manifest_sha256": sha256_file(authority_path),
        "recording_count": EXPECTED_RECORDINGS,
        "speaker_clip_count": EXPECTED_SPEAKERS,
    }


def _public_audio_manifest(clips: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "slot_id": clip["slot_id"],
            "document_id": clip["document_id"],
            "speaker_ref": clip["speaker_ref"],
            "relative_path": clip["relative_path"],
            "sha256": clip["sha256"],
            "bytes": clip["bytes"],
        }
        for clip in clips
    ]


def _copy_private_audio(
    clip: Mapping[str, Any], *, destination: Path, runtime_root: Path, acoustic_root: Path
) -> None:
    source = Path(clip["source_path"])
    require_private_file(source, acoustic_root)
    ensure_private_tree(runtime_root, destination.parent)
    if destination.exists():
        require_private_file(destination, runtime_root)
        if (
            destination.stat().st_size != clip["bytes"]
            or sha256_file(destination) != clip["sha256"]
        ):
            _fail("A copied review clip changed in place.")
        return
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with source.open("rb") as input_stream, os.fdopen(descriptor, "wb") as output_stream:
        shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
        output_stream.flush()
        os.fsync(output_stream.fileno())
    if (
        sha256_file(source) != clip["sha256"]
        or destination.stat().st_size != clip["bytes"]
        or sha256_file(destination) != clip["sha256"]
    ):
        _fail("A review clip changed during its bounded private copy.")


def _write_private_text(path: Path, content: str, root: Path) -> None:
    encoded = content.encode("utf-8")
    expected = hashlib.sha256(encoded).hexdigest()
    if path.exists():
        require_private_file(path, root)
        if sha256_file(path) != expected:
            _fail("The private review worksheet changed in place.")
        return
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def prepare_live_worksheet(
    *,
    plan0060_root: Path = DEFAULT_PLAN0060_ROOT,
    acoustic_root: Path = DEFAULT_PLAN0060_ACOUSTIC_ROOT,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    live_store_root: Path = DEFAULT_LIVE_STORE_ROOT,
) -> dict[str, Any]:
    """Freeze or replay one private minimum-copy, non-applying worksheet."""

    repository = _repository_authority()
    source, bindings = _validated_live_source(
        plan0060_root=plan0060_root, live_store_root=live_store_root
    )
    audio_sources, acoustic_binding = _validated_audio_sources(source, acoustic_root)
    bindings = {**bindings, "acoustic_clips": acoustic_binding}
    audio_manifest = _public_audio_manifest(audio_sources)
    worksheet = render_review_worksheet(source)
    worksheet_sha256 = hashlib.sha256(worksheet.encode("utf-8")).hexdigest()
    paths = _worksheet_paths(runtime_root, worksheet_sha256)
    if paths["receipt"].exists():
        return replay_live_worksheet(
            worksheet_sha256,
            plan0060_root=plan0060_root,
            acoustic_root=acoustic_root,
            runtime_root=runtime_root,
            live_store_root=live_store_root,
        )
    if paths["run"].exists():
        _fail("A partial Plan 0061 worksheet directory already exists.")
    ensure_private_tree(paths["root"], paths["run"])
    ensure_private_tree(paths["root"], paths["bundle"])
    _write_private_text(paths["worksheet"], worksheet, paths["root"])
    for clip in audio_sources:
        _copy_private_audio(
            clip,
            destination=paths["bundle"] / str(clip["relative_path"]),
            runtime_root=paths["root"],
            acoustic_root=acoustic_root.expanduser().absolute(),
        )
    manifest = {
        "schema_version": WORKSHEET_MANIFEST_SCHEMA,
        "status": "human_review_worksheet_prepared",
        "worksheet_schema_version": WORKSHEET_SCHEMA,
        "worksheet_sha256": worksheet_sha256,
        "source_bindings": bindings,
        "repository_authority": repository,
        "recording_count": EXPECTED_RECORDINGS,
        "speaker_slot_count": EXPECTED_SPEAKERS,
        "condition_count": EXPECTED_CONDITIONS,
        "preselected_decision_count": 0,
        "human_decision_count": 0,
        "apply_enabled": False,
        "contains_private_audio_clips": True,
        "contains_full_recording_audio": False,
        "audio_clip_count": EXPECTED_SPEAKERS,
        "audio_total_bytes": sum(clip["bytes"] for clip in audio_manifest),
        "audio_clips": audio_manifest,
        "contains_raw_transcript": False,
        "contains_candidate_labels": True,
        "contains_candidate_email": False,
        "negative_actions": NEGATIVE_ACTIONS,
    }
    write_immutable_private_json(paths["manifest"], manifest)
    receipt_core = {
        "schema_version": WORKSHEET_RECEIPT_SCHEMA,
        "status": "human_review_worksheet_prepared",
        "worksheet_sha256": worksheet_sha256,
        "manifest_sha256": sha256_file(paths["manifest"]),
        "p4_content_sha256": PLAN0060_P4_CONTENT_SHA256,
        "p4_manifest_sha256": PLAN0060_P4_MANIFEST_SHA256,
        "recording_count": EXPECTED_RECORDINGS,
        "speaker_slot_count": EXPECTED_SPEAKERS,
        "condition_count": EXPECTED_CONDITIONS,
        "preselected_decision_count": 0,
        "human_decision_count": 0,
        "apply_enabled": False,
        "contains_private_audio_clips": True,
        "contains_full_recording_audio": False,
        "audio_clip_count": EXPECTED_SPEAKERS,
        "audio_total_bytes": sum(clip["bytes"] for clip in audio_manifest),
        "contains_raw_transcript": False,
        "live_mutation_count": 0,
        "mode": "0600",
    }
    receipt = {**receipt_core, "content_sha256": canonical_artifact_hash(receipt_core)}
    write_immutable_private_json(paths["receipt"], receipt)
    return {
        **receipt,
        "worksheet_path": str(paths["worksheet"]),
        "bundle_path": str(paths["bundle"]),
        "manifest_path": str(paths["manifest"]),
        "idempotent_replay": False,
    }


def replay_live_worksheet(
    worksheet_sha256: str,
    *,
    plan0060_root: Path = DEFAULT_PLAN0060_ROOT,
    acoustic_root: Path = DEFAULT_PLAN0060_ACOUSTIC_ROOT,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    live_store_root: Path = DEFAULT_LIVE_STORE_ROOT,
) -> dict[str, Any]:
    selected_sha256 = _sha256(worksheet_sha256, field="worksheet SHA-256")
    paths = _worksheet_paths(runtime_root, selected_sha256)
    for key in ("worksheet", "manifest", "receipt"):
        require_private_file(paths[key], paths["root"])
    manifest = read_private_object(paths["manifest"])
    receipt = read_private_object(paths["receipt"])
    source, bindings = _validated_live_source(
        plan0060_root=plan0060_root, live_store_root=live_store_root
    )
    audio_sources, acoustic_binding = _validated_audio_sources(source, acoustic_root)
    bindings = {**bindings, "acoustic_clips": acoustic_binding}
    audio_manifest = _public_audio_manifest(audio_sources)
    for clip in audio_sources:
        destination = paths["bundle"] / str(clip["relative_path"])
        require_private_file(destination, paths["root"])
        if (
            destination.stat().st_size != clip["bytes"]
            or sha256_file(destination) != clip["sha256"]
        ):
            _fail("A frozen Plan 0061 review clip drifted.")
    expected_worksheet = render_review_worksheet(source)
    expected_manifest = {
        **manifest,
        "source_bindings": bindings,
        "audio_clips": audio_manifest,
    }
    receipt_core = {key: value for key, value in receipt.items() if key != "content_sha256"}
    if (
        sha256_file(paths["worksheet"]) != selected_sha256
        or hashlib.sha256(expected_worksheet.encode("utf-8")).hexdigest()
        != selected_sha256
        or manifest != expected_manifest
        or manifest.get("schema_version") != WORKSHEET_MANIFEST_SCHEMA
        or manifest.get("worksheet_sha256") != selected_sha256
        or manifest.get("preselected_decision_count") != 0
        or manifest.get("human_decision_count") != 0
        or manifest.get("apply_enabled") is not False
        or manifest.get("negative_actions") != NEGATIVE_ACTIONS
        or manifest.get("contains_private_audio_clips") is not True
        or manifest.get("contains_full_recording_audio") is not False
        or manifest.get("audio_clip_count") != EXPECTED_SPEAKERS
        or manifest.get("audio_total_bytes")
        != sum(clip["bytes"] for clip in audio_manifest)
        or receipt.get("schema_version") != WORKSHEET_RECEIPT_SCHEMA
        or receipt.get("worksheet_sha256") != selected_sha256
        or receipt.get("manifest_sha256") != sha256_file(paths["manifest"])
        or receipt.get("contains_private_audio_clips") is not True
        or receipt.get("contains_full_recording_audio") is not False
        or receipt.get("audio_clip_count") != EXPECTED_SPEAKERS
        or receipt.get("content_sha256") != canonical_artifact_hash(receipt_core)
    ):
        _fail("The frozen Plan 0061 worksheet evidence drifted.")
    return {
        **receipt,
        "worksheet_path": str(paths["worksheet"]),
        "bundle_path": str(paths["bundle"]),
        "manifest_path": str(paths["manifest"]),
        "idempotent_replay": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare or validate the non-applying Plan 0061 review worksheet."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--plan0060-root", type=Path, default=DEFAULT_PLAN0060_ROOT)
    prepare.add_argument(
        "--acoustic-root", type=Path, default=DEFAULT_PLAN0060_ACOUSTIC_ROOT
    )
    prepare.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    prepare.add_argument("--live-store-root", type=Path, default=DEFAULT_LIVE_STORE_ROOT)
    replay = subparsers.add_parser("replay")
    replay.add_argument("--worksheet-sha256", required=True)
    replay.add_argument("--plan0060-root", type=Path, default=DEFAULT_PLAN0060_ROOT)
    replay.add_argument(
        "--acoustic-root", type=Path, default=DEFAULT_PLAN0060_ACOUSTIC_ROOT
    )
    replay.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    replay.add_argument("--live-store-root", type=Path, default=DEFAULT_LIVE_STORE_ROOT)
    validate = subparsers.add_parser("validate-answers")
    validate.add_argument("--answers-file", type=Path, required=True)
    validate.add_argument("--plan0060-root", type=Path, default=DEFAULT_PLAN0060_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare_live_worksheet(
                plan0060_root=args.plan0060_root,
                acoustic_root=args.acoustic_root,
                runtime_root=args.runtime_root,
                live_store_root=args.live_store_root,
            )
        elif args.command == "replay":
            result = replay_live_worksheet(
                args.worksheet_sha256,
                plan0060_root=args.plan0060_root,
                acoustic_root=args.acoustic_root,
                runtime_root=args.runtime_root,
                live_store_root=args.live_store_root,
            )
        else:
            root = args.plan0060_root.expanduser().absolute()
            receipt = plan0060.replay_review_packet(
                runtime_root=root,
                activation_sha256=PLAN0060_ACTIVATION_SHA256,
            )
            manifest_path = Path(str(receipt["manifest_path"]))
            require_private_file(manifest_path, root)
            answer_path = args.answers_file.expanduser().absolute()
            if not answer_path.is_file() or answer_path.is_symlink() or answer_path.stat().st_mode & 0o077:
                _fail("The answer file must be a private 0600 regular file.")
            result = parse_decision_block(
                answer_path.read_text(encoding="utf-8"),
                read_private_object(manifest_path),
            )
        print(json.dumps(result, sort_keys=True))
        return 0
    except (OSError, sqlite3.Error, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
