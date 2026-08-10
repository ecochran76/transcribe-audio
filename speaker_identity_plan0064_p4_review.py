"""Plan 0064 P4 private direct-audio gold review surface."""

from __future__ import annotations

import argparse
from array import array
import html
import json
from pathlib import Path
import struct
from typing import Any, Mapping, Sequence
import wave

from acoustic_audio_derivatives import (
    canonical_artifact_hash,
    ensure_private_tree,
    read_private_object,
    require_private_file,
    sha256_file,
    write_immutable_private_json,
)
from speaker_identity_plan0064_p0 import DEFAULT_RUNTIME_ROOT
from speaker_identity_plan0064_p1 import SAMPLE_RATE, _decode, _slot_probe
from speaker_identity_plan0064_p2 import _phase_safe_p0
from speaker_identity_plan0064_p3 import replay_p3


PREVIEW_SCHEMA = "transcribe-audio.plan0064-p4-review-preview.v1"
AUTHORITY_SCHEMA = "transcribe-audio.plan0064-p4-review-authority.v1"
RECEIPT_SCHEMA = "transcribe-audio.plan0064-p4-review-receipt.v1"
DECISION_SCHEMA = "transcribe-audio.plan0064-p4-human-gold-decisions.v1"
ACTION_COUNTS = {
    "speaker_assignments": 0,
    "new_enrollments": 0,
    "profile_mutations": 0,
    "knowledge_writes": 0,
    "provider_writes": 0,
    "external_provider_writes": 0,
}


class Plan0064P4ReviewError(ValueError):
    """Raised when the P4 review denominator or private artifacts drift."""


def _hash(value: Any) -> str:
    return canonical_artifact_hash(value)


def _content_addressed(value: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(value)
    body.pop("content_sha256", None)
    return {**body, "content_sha256": _hash(body)}


def _read(path: Path) -> dict[str, Any]:
    value = read_private_object(path)
    if not isinstance(value, dict):
        raise Plan0064P4ReviewError(f"Private artifact is not an object: {path}")
    return value


def _authority_inputs(
    p0_content_sha256: str, *, runtime_root: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    p0, _bridge = _phase_safe_p0(p0_content_sha256, runtime_root=runtime_root)
    manifest = _read(Path(p0["private_manifest_path"]))
    p3_receipt = replay_p3(p0_content_sha256, runtime_root=runtime_root)
    resolution = _read(Path(p3_receipt["private_resolution_path"]))
    return manifest, p3_receipt, resolution


def _people(manifest: Mapping[str, Any]) -> list[dict[str, str]]:
    people = [
        {
            "person_id": str(item.get("person_id") or ""),
            "display_name": str(item.get("primary_name") or "").strip(),
        }
        for item in manifest["canonical_bindings"]["current_person_profiles"]
        if str(item.get("person_id") or "")
        and str(item.get("primary_name") or "").strip()
        and str(item.get("resolution_status") or "") == "reviewed"
    ]
    if len(people) != len({item["person_id"] for item in people}) or not people:
        raise Plan0064P4ReviewError("Canonical review options are incomplete or repeated.")
    return sorted(people, key=lambda item: (item["display_name"].casefold(), item["person_id"]))


def build_p4_review_preview(
    p0_content_sha256: str, *, runtime_root: Path = DEFAULT_RUNTIME_ROOT
) -> dict[str, Any]:
    manifest, p3_receipt, resolution = _authority_inputs(
        p0_content_sha256, runtime_root=runtime_root
    )
    selected = [
        item
        for item in manifest["evaluation_cohort"]["considered"]
        if item["disposition"] == "selected_evaluation_candidate"
    ]
    slots = [
        slot
        for recording in resolution["recordings"]
        for slot in recording["speaker_slots"]
    ]
    if (
        len(selected) != 12
        or len(slots) != 39
        or [item["document_id"] for item in selected]
        != [item["document_id"] for item in resolution["recordings"]]
    ):
        raise Plan0064P4ReviewError("P4 review denominator differs from P0/P3.")
    people = _people(manifest)
    return _content_addressed(
        {
            "schema_version": PREVIEW_SCHEMA,
            "status": "ready_for_private_direct_audio_review",
            "p0_content_sha256": p0_content_sha256,
            "p3_receipt_content_sha256": p3_receipt["content_sha256"],
            "p3_resolution_content_sha256": resolution["content_sha256"],
            "recording_count": len(selected),
            "speaker_slot_count": len(slots),
            "canonical_option_count": len(people),
            "decision_options": ["canonical_person", "not_listed", "unresolved"],
            "model_predictions_visible": False,
            "contains_private_audio": True,
            "human_decision_count": 0,
            "will_apply_speaker_identity": False,
            "action_counts": dict(ACTION_COUNTS),
        }
    )


def _write_clip(path: Path, samples: array) -> None:
    if not samples:
        raise Plan0064P4ReviewError("A review speaker has no playable audio.")
    bounded = [max(-32768, min(32767, round(float(value) * 32767))) for value in samples]
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(SAMPLE_RATE)
        audio.writeframes(struct.pack(f"<{len(bounded)}h", *bounded))
    path.chmod(0o600)


def build_review_html(
    *, authority_content_sha256: str, cases: Sequence[Mapping[str, Any]],
    people: Sequence[Mapping[str, str]],
) -> str:
    option_html = "".join(
        f'<option value="person:{html.escape(item["person_id"])}">'
        f'{html.escape(item["display_name"])}</option>'
        for item in people
    )
    cards = []
    for index, case in enumerate(cases, start=1):
        cards.append(
            "".join(
                [
                    f'<article class="card" data-speaker-ref="{html.escape(case["speaker_ref"])}">',
                    f'<div class="ordinal">Speaker {index} of {len(cases)}</div>',
                    f'<h2>{html.escape(case["recording_label"])} · {html.escape(case["speaker_label"])}</h2>',
                    '<p>Who is speaking in this direct audio sample?</p>',
                    f'<audio controls preload="metadata" src="{html.escape(case["clip_relative_path"])}"></audio>',
                    '<label>Identity decision<select class="decision">',
                    '<option value="">Choose one…</option>',
                    option_html,
                    '<option value="not_listed">Not listed</option>',
                    '<option value="unresolved">Unresolved</option>',
                    '</select></label>',
                    '<label>Optional note<textarea class="note" maxlength="300"></textarea></label>',
                    '</article>',
                ]
            )
        )
    script = f"""
const authority = {json.dumps(authority_content_sha256)};
const schema = {json.dumps(DECISION_SCHEMA)};
const cards = Array.from(document.querySelectorAll('.card'));
function decisions() {{
  return cards.map((card) => {{
    const raw = card.querySelector('.decision').value;
    return {{
      speaker_ref: card.dataset.speakerRef,
      decision: raw.startsWith('person:') ? 'canonical_person' : raw,
      person_id: raw.startsWith('person:') ? raw.slice(7) : null,
      note: card.querySelector('.note').value.trim()
    }};
  }});
}}
function render() {{
  const rows = decisions();
  const complete = rows.filter((row) => row.decision).length;
  document.getElementById('progress').textContent = `${{complete}} / ${{rows.length}} complete`;
  document.getElementById('export').disabled = complete !== rows.length;
}}
document.addEventListener('change', render);
document.getElementById('export').addEventListener('click', async () => {{
  const payload = JSON.stringify({{
    schema_version: schema,
    authority_content_sha256: authority,
    decisions: decisions()
  }}, null, 2);
  document.getElementById('output').value = payload;
  try {{ await navigator.clipboard.writeText(payload); }} catch (_error) {{}}
}});
render();
"""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Plan 0064 direct-audio review</title>
<style>
:root{{--ink:#18212f;--muted:#5c6878;--paper:#f4f1ea;--card:#fff;--accent:#234f5f;}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.45 system-ui,sans-serif}}
main{{max-width:980px;margin:auto;padding:28px}} header{{margin-bottom:24px}} h1{{margin:.2rem 0}} .muted,.ordinal{{color:var(--muted)}}
.toolbar{{position:sticky;top:0;z-index:2;background:#f4f1eaf2;padding:12px 0;display:flex;gap:16px;align-items:center}}
.card{{background:var(--card);border:1px solid #d8d2c7;border-radius:14px;padding:20px;margin:16px 0;box-shadow:0 4px 14px #0000000b}}
.card h2{{margin:.25rem 0}} audio,select,textarea{{display:block;width:100%;margin:8px 0 16px}} select,textarea,button{{font:inherit;padding:10px;border-radius:8px;border:1px solid #9aa3ad}}
textarea{{min-height:64px}} button{{background:var(--accent);color:#fff;border:0;font-weight:700}} button:disabled{{opacity:.45}}
#output{{min-height:220px;font-family:ui-monospace,monospace}} @media(max-width:600px){{main{{padding:16px}}}}
</style></head><body><main>
<header><div class="muted">Private human-gold gate · no model predictions shown</div><h1>Plan 0064 direct-audio review</h1>
<p>Listen to each speaker-specific clip and choose the literal identity. Complete all 39 rows before exporting.</p></header>
<div class="toolbar"><strong id="progress"></strong><button id="export" disabled>Copy complete JSON</button></div>
{''.join(cards)}
<section><h2>Complete decision export</h2><p>After all rows are complete, copy this JSON back to the agent.</p><textarea id="output" readonly></textarea></section>
</main><script>{script}</script></body></html>"""


def execute_p4_review(
    p0_content_sha256: str, *, runtime_root: Path = DEFAULT_RUNTIME_ROOT
) -> dict[str, Any]:
    preview = build_p4_review_preview(p0_content_sha256, runtime_root=runtime_root)
    manifest, p3_receipt, resolution = _authority_inputs(
        p0_content_sha256, runtime_root=runtime_root
    )
    root = runtime_root.expanduser().absolute() / f"p4-review-{preview['content_sha256'][:24]}"
    clips_root = root / "clips"
    authority_path, html_path = root / "review-authority.json", root / "index.html"
    template_path, receipt_path = root / "decision-template.json", root / "receipt.json"
    if root.exists():
        return replay_p4_review(p0_content_sha256, runtime_root=runtime_root)
    ensure_private_tree(root, clips_root)
    selected = [
        item
        for item in manifest["evaluation_cohort"]["considered"]
        if item["disposition"] == "selected_evaluation_candidate"
    ]
    cases = []
    for recording_index, recording in enumerate(selected, start=1):
        transcript_path = Path(recording["transcript_artifact"]["path"])
        media_path = Path(recording["source_media_artifact"]["path"])
        transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
        samples = _decode(media_path)
        for speaker_index, speaker in enumerate(recording["speaker_labels"], start=1):
            speaker_ref = f"{recording['document_id']}::{speaker}"
            clip_name = f"r{recording_index:02d}-s{speaker_index:02d}-{_hash(speaker_ref)[:12]}.wav"
            clip_path = clips_root / clip_name
            _write_clip(clip_path, _slot_probe(transcript, speaker, samples))
            cases.append(
                {
                    "speaker_ref": speaker_ref,
                    "document_id": recording["document_id"],
                    "speaker_label": speaker,
                    "recording_label": f"Recording {recording_index} of {len(selected)}",
                    "clip_relative_path": f"clips/{clip_name}",
                    "clip_sha256": sha256_file(clip_path),
                }
            )
    if len(cases) != preview["speaker_slot_count"]:
        raise Plan0064P4ReviewError("Generated review clips do not cover every slot.")
    people = _people(manifest)
    authority = _content_addressed(
        {
            "schema_version": AUTHORITY_SCHEMA,
            "status": "awaiting_complete_literal_human_gold",
            "preview_content_sha256": preview["content_sha256"],
            "p3_receipt_content_sha256": p3_receipt["content_sha256"],
            "p3_resolution_content_sha256": resolution["content_sha256"],
            "cases": cases,
            "people": people,
            "human_decision_count": 0,
            "model_predictions_visible": False,
            "action_counts": dict(ACTION_COUNTS),
        }
    )
    write_immutable_private_json(authority_path, authority)
    template = {
        "schema_version": DECISION_SCHEMA,
        "authority_content_sha256": authority["content_sha256"],
        "decisions": [
            {"speaker_ref": item["speaker_ref"], "decision": "", "person_id": None, "note": ""}
            for item in cases
        ],
    }
    write_immutable_private_json(template_path, template)
    html_path.write_text(
        build_review_html(
            authority_content_sha256=authority["content_sha256"],
            cases=cases,
            people=people,
        ),
        encoding="utf-8",
    )
    html_path.chmod(0o600)
    receipt = _content_addressed(
        {
            "schema_version": RECEIPT_SCHEMA,
            "status": "p4_private_review_ready",
            "preview_content_sha256": preview["content_sha256"],
            "authority_content_sha256": authority["content_sha256"],
            "authority_file_sha256": sha256_file(authority_path),
            "html_file_sha256": sha256_file(html_path),
            "template_file_sha256": sha256_file(template_path),
            "clip_set_sha256": _hash([item["clip_sha256"] for item in cases]),
            "recording_count": preview["recording_count"],
            "speaker_slot_count": len(cases),
            "human_decision_count": 0,
            "action_counts": dict(ACTION_COUNTS),
        }
    )
    write_immutable_private_json(receipt_path, receipt)
    return {**receipt, "private_review_root": str(root), "private_html_path": str(html_path), "idempotent_replay": False}


def replay_p4_review(
    p0_content_sha256: str, *, runtime_root: Path = DEFAULT_RUNTIME_ROOT
) -> dict[str, Any]:
    preview = build_p4_review_preview(p0_content_sha256, runtime_root=runtime_root)
    root = runtime_root.expanduser().absolute() / f"p4-review-{preview['content_sha256'][:24]}"
    authority_path, html_path = root / "review-authority.json", root / "index.html"
    template_path, receipt_path = root / "decision-template.json", root / "receipt.json"
    for path in (authority_path, html_path, template_path, receipt_path):
        require_private_file(path, root)
    authority, receipt = _read(authority_path), _read(receipt_path)
    clips = [root / item["clip_relative_path"] for item in authority["cases"]]
    for path in clips:
        require_private_file(path, root)
    if (
        authority.get("content_sha256")
        != _hash({key: value for key, value in authority.items() if key != "content_sha256"})
        or receipt.get("preview_content_sha256") != preview["content_sha256"]
        or receipt.get("authority_content_sha256") != authority["content_sha256"]
        or receipt.get("authority_file_sha256") != sha256_file(authority_path)
        or receipt.get("html_file_sha256") != sha256_file(html_path)
        or receipt.get("template_file_sha256") != sha256_file(template_path)
        or receipt.get("clip_set_sha256") != _hash([sha256_file(path) for path in clips])
        or receipt.get("content_sha256")
        != _hash({key: value for key, value in receipt.items() if key != "content_sha256"})
        or receipt.get("action_counts") != ACTION_COUNTS
    ):
        raise Plan0064P4ReviewError("The private P4 review artifact drifted.")
    return {**receipt, "private_review_root": str(root), "private_html_path": str(html_path), "idempotent_replay": True}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preview", "execute", "replay"))
    parser.add_argument("--p0-content-sha256", required=True)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    args = parser.parse_args(argv)
    action = {"preview": build_p4_review_preview, "execute": execute_p4_review, "replay": replay_p4_review}[args.action]
    print(json.dumps(action(args.p0_content_sha256, runtime_root=args.runtime_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
