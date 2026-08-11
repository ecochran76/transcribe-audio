"""Publish Plan 0071 D2's blinded filename-bearing direct-audio review."""

from __future__ import annotations

import hashlib
import html
import json
import struct
import subprocess
import wave
from array import array
from pathlib import Path
from typing import Any, Mapping, Sequence

import speaker_identity_plan0064_p1 as plan0064_p1
import speaker_identity_plan0064_p4_review as plan0064_p4
import speaker_identity_plan0071_d2_predictions as predictions
import speaker_identity_plan0071_d2_predictions_attempt2 as attempt2
from acoustic_audio_derivatives import (
    ensure_private_tree,
    read_private_object,
    require_private_file,
    sha256_file,
    write_immutable_private_json,
)


PREVIEW_SCHEMA = "transcribe-audio.plan0071-d2-review-preview.v1"
AUTHORITY_SCHEMA = "transcribe-audio.plan0071-d2-review-authority.v1"
RECEIPT_SCHEMA = "transcribe-audio.plan0071-d2-review-receipt.v1"
DECISION_SCHEMA = "transcribe-audio.plan0071-d2-human-gold-decisions.v1"
DEFAULT_RUNTIME_ROOT = attempt2.DEFAULT_RUNTIME_ROOT
PREDICTION_RECEIPT_CONTENT_SHA256 = (
    "8de26c83af3a2dc1da7c04633fad4c698adcccf3972d42d56f7a8aecf86971b6"
)
PREDICTION_MANIFEST_CONTENT_SHA256 = (
    "71fc568512b5a0c24319445df3ffe0bdbc89957b94005cd46af6acc6b182ffd2"
)
PREDICTION_RESOLUTION_CONTENT_SHA256 = (
    "bf1876e0610f668ea8eaa4f5a0c4f3748540df36523e39b4410eb8428ebfe931"
)
MUTATION_EFFECT_COUNTS = dict(attempt2.MUTATION_EFFECT_COUNTS)


class Plan0071D2ReviewError(ValueError):
    """Raised when the blinded D2 review authority or assets drift."""


def _hash(value: Any) -> str:
    return predictions._hash(value)


def _content(value: Mapping[str, Any]) -> dict[str, Any]:
    return predictions._content(value)


def _validate_content(value: Mapping[str, Any], label: str) -> None:
    try:
        predictions._validate_content(value, label)
    except predictions.Plan0071D2PredictionError as exc:
        raise Plan0071D2ReviewError(str(exc)) from exc


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], check=False, capture_output=True, text=True
    )
    if result.returncode:
        raise Plan0071D2ReviewError(
            result.stderr.strip() or "Git authority read failed."
        )
    return result.stdout.strip()


def _source_authority(*, require_clean: bool) -> dict[str, Any]:
    module = Path(__file__).resolve()
    root = Path(_git("rev-parse", "--show-toplevel")).resolve()
    relative = module.relative_to(root).as_posix()
    commit = _git("log", "-1", "--format=%H", "--", relative)
    committed = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        check=False,
        capture_output=True,
    )
    upstream = _git("rev-parse", "@{upstream}")
    module_sha256 = hashlib.sha256(module.read_bytes()).hexdigest()
    value = {
        "module_name": relative,
        "module_commit": commit,
        "module_sha256": module_sha256,
        "module_blob_matches": (
            committed.returncode == 0
            and module_sha256 == hashlib.sha256(committed.stdout).hexdigest()
        ),
        "clean": not _git("status", "--porcelain=v1"),
        "upstream_ahead": int(_git("rev-list", "--count", f"{upstream}..HEAD")),
        "upstream_behind": int(_git("rev-list", "--count", f"HEAD..{upstream}")),
    }
    if value["module_blob_matches"] is not True or (
        require_clean
        and (
            value["clean"] is not True
            or value["upstream_ahead"]
            or value["upstream_behind"]
        )
    ):
        raise Plan0071D2ReviewError(
            "D2 review source is not committed, clean, and upstream-even."
        )
    return value


def _prediction_authority(runtime_root: Path) -> dict[str, Any]:
    replay = attempt2.replay_attempt2(runtime_root=runtime_root)
    paths = attempt2._paths(runtime_root)
    manifest = read_private_object(paths["manifest"])
    resolution = read_private_object(paths["resolution"])
    _validate_content(manifest, "D2 attempt-2 prediction manifest")
    _validate_content(resolution, "D2 attempt-2 prediction resolution")
    if (
        replay.get("content_sha256") != PREDICTION_RECEIPT_CONTENT_SHA256
        or manifest.get("content_sha256") != PREDICTION_MANIFEST_CONTENT_SHA256
        or resolution.get("content_sha256")
        != PREDICTION_RESOLUTION_CONTENT_SHA256
        or manifest.get("human_gold_read") is not False
        or manifest.get("execution_counts", {}).get("capture_evaluation_calls") != 0
        or manifest.get("mutation_effect_counts") != MUTATION_EFFECT_COUNTS
    ):
        raise Plan0071D2ReviewError("The D2 prediction authority drifted.")
    inherited = predictions._bound_authorities(runtime_root)
    selected = predictions._selected(inherited["cohort_manifest"])
    people = plan0064_p4._people(
        read_private_object(Path(inherited["p0_binding"]["path"]))
    )
    if len(people) != 6:
        raise Plan0071D2ReviewError("The canonical decision roster drifted.")
    return {
        "replay": replay,
        "manifest": manifest,
        "resolution": resolution,
        "selected": selected,
        "people": people,
    }


def build_review_preview(
    *, runtime_root: Path = DEFAULT_RUNTIME_ROOT
) -> dict[str, Any]:
    authority = _prediction_authority(runtime_root)
    selected = authority["selected"]
    filenames = [str(item["original_recording_filename"]) for item in selected]
    return _content(
        {
            "schema_version": PREVIEW_SCHEMA,
            "status": "ready_for_private_direct_audio_review",
            "prediction_receipt_content_sha256": PREDICTION_RECEIPT_CONTENT_SHA256,
            "prediction_manifest_content_sha256": PREDICTION_MANIFEST_CONTENT_SHA256,
            "prediction_resolution_content_sha256": (
                PREDICTION_RESOLUTION_CONTENT_SHA256
            ),
            "recording_count": len(selected),
            "speaker_slot_count": sum(
                len(item.get("speaker_labels") or []) for item in selected
            ),
            "canonical_option_count": len(authority["people"]),
            "original_recording_filename_count": len(filenames),
            "original_recording_filename_set_sha256": _hash(filenames),
            "decision_options": ["canonical_person", "not_listed", "unresolved"],
            "model_predictions_visible": False,
            "contains_private_audio": True,
            "human_decision_count": 0,
            "fresh_evaluation_allowed": False,
            "mutation_effect_counts": dict(MUTATION_EFFECT_COUNTS),
        }
    )


def _write_clip(path: Path, samples: array) -> None:
    if not samples:
        raise Plan0071D2ReviewError("A review speaker has no playable audio.")
    bounded = [max(-32768, min(32767, round(float(value) * 32767))) for value in samples]
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(plan0064_p1.SAMPLE_RATE)
        audio.writeframes(struct.pack(f"<{len(bounded)}h", *bounded))
    path.chmod(0o600)


def build_review_html(
    *,
    authority_content_sha256: str,
    cases: Sequence[Mapping[str, Any]],
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
                    f'<h2>{html.escape(case["recording_label"])} · Speaker {html.escape(case["speaker_label"])}</h2>',
                    '<p class="filename"><strong>Original recording:</strong> <code>',
                    html.escape(case["original_recording_filename"]),
                    '</code></p>',
                    '<p>Who is speaking in this direct audio sample?</p>',
                    f'<audio controls preload="metadata" src="{html.escape(case["clip_relative_path"])}"></audio>',
                    '<label>Identity decision<select class="decision">',
                    '<option value="">Choose one…</option>',
                    option_html,
                    '<option value="not_listed">Not listed</option>',
                    '<option value="unresolved">Unresolved</option>',
                    '</select></label>',
                    '<label>Optional name or note<textarea class="note" maxlength="300"></textarea></label>',
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
<title>Plan 0071 supplemental direct-audio review</title>
<style>
:root{{--ink:#18212f;--muted:#5c6878;--paper:#f4f1ea;--card:#fff;--accent:#234f5f;}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.45 system-ui,sans-serif}}
main{{max-width:980px;margin:auto;padding:28px}} header{{margin-bottom:24px}} h1{{margin:.2rem 0}} .muted,.ordinal{{color:var(--muted)}}
.toolbar{{position:sticky;top:0;z-index:2;background:#f4f1eaf2;padding:12px 0;display:flex;gap:16px;align-items:center}}
.card{{background:var(--card);border:1px solid #d8d2c7;border-radius:14px;padding:20px;margin:16px 0;box-shadow:0 4px 14px #0000000b}}
.card h2{{margin:.25rem 0}} .filename{{margin:.5rem 0 1rem;color:var(--muted)}} .filename code{{color:var(--ink);overflow-wrap:anywhere}} audio,select,textarea{{display:block;width:100%;margin:8px 0 16px}} select,textarea,button{{font:inherit;padding:10px;border-radius:8px;border:1px solid #9aa3ad}}
textarea{{min-height:64px}} button{{background:var(--accent);color:#fff;border:0;font-weight:700}} button:disabled{{opacity:.45}}
#output{{min-height:220px;font-family:ui-monospace,monospace}} @media(max-width:600px){{main{{padding:16px}}}}
</style></head><body><main>
<header><div class="muted">Private literal human-gold gate · no model predictions shown</div><h1>Plan 0071 supplemental direct-audio review</h1>
<p>Listen to each speaker-specific clip and choose the literal identity. The original recording filename is shown on every card. Complete all 18 rows before exporting.</p></header>
<div class="toolbar"><strong id="progress"></strong><button id="export" disabled>Copy complete JSON</button></div>
{''.join(cards)}
<section><h2>Complete decision export</h2><p>After all rows are complete, copy this JSON back to the agent.</p><textarea id="output" readonly></textarea></section>
</main><script>{script}</script></body></html>"""


def _paths(runtime_root: Path, preview_content_sha256: str) -> dict[str, Path]:
    root = runtime_root.expanduser().resolve()
    run = root / f"d2-review-{preview_content_sha256[:24]}"
    return {
        "root": root,
        "run": run,
        "clips": run / "clips",
        "authority": run / "review-authority.json",
        "html": run / "index.html",
        "template": run / "decision-template.json",
        "receipt": run / "receipt.json",
    }


def execute_review(*, runtime_root: Path = DEFAULT_RUNTIME_ROOT) -> dict[str, Any]:
    preview = build_review_preview(runtime_root=runtime_root)
    paths = _paths(runtime_root, preview["content_sha256"])
    if paths["receipt"].exists():
        return replay_review(runtime_root=runtime_root)
    source_authority = _source_authority(require_clean=True)
    bound = _prediction_authority(runtime_root)
    ensure_private_tree(paths["root"], paths["clips"])
    cases = []
    selected = bound["selected"]
    for recording_index, recording in enumerate(selected, start=1):
        transcript_path = Path(str(recording["transcript_artifact"]["path"]))
        media_path = Path(str(recording["source_media_artifact"]["path"]))
        if (
            sha256_file(transcript_path) != recording["transcript_sha256"]
            or sha256_file(media_path) != recording["source_media_sha256"]
        ):
            raise Plan0071D2ReviewError("A D2 review source artifact drifted.")
        transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
        samples = plan0064_p1._decode(media_path)
        for speaker_index, speaker in enumerate(
            recording["speaker_labels"], start=1
        ):
            speaker_ref = f"{recording['document_id']}::{speaker}"
            clip_name = (
                f"r{recording_index:02d}-s{speaker_index:02d}-"
                f"{_hash(speaker_ref)[:12]}.wav"
            )
            clip_path = paths["clips"] / clip_name
            _write_clip(
                clip_path,
                plan0064_p1._slot_probe(transcript, str(speaker), samples),
            )
            cases.append(
                {
                    "speaker_ref": speaker_ref,
                    "document_id": recording["document_id"],
                    "speaker_label": speaker,
                    "recording_label": (
                        f"Recording {recording_index} of {len(selected)}"
                    ),
                    "original_recording_filename": recording[
                        "original_recording_filename"
                    ],
                    "clip_relative_path": f"clips/{clip_name}",
                    "clip_sha256": sha256_file(clip_path),
                }
            )
    if len(cases) != 18:
        raise Plan0071D2ReviewError("Review clips do not cover all 18 slots.")
    authority = _content(
        {
            "schema_version": AUTHORITY_SCHEMA,
            "status": "awaiting_complete_literal_human_gold",
            "source_authority": source_authority,
            "preview_content_sha256": preview["content_sha256"],
            "prediction_receipt_content_sha256": (
                PREDICTION_RECEIPT_CONTENT_SHA256
            ),
            "prediction_resolution_content_sha256": (
                PREDICTION_RESOLUTION_CONTENT_SHA256
            ),
            "original_recording_filename_set_sha256": preview[
                "original_recording_filename_set_sha256"
            ],
            "cases": cases,
            "people": bound["people"],
            "human_decision_count": 0,
            "model_predictions_visible": False,
            "fresh_evaluation_allowed": False,
            "mutation_effect_counts": dict(MUTATION_EFFECT_COUNTS),
        }
    )
    write_immutable_private_json(paths["authority"], authority)
    template = {
        "schema_version": DECISION_SCHEMA,
        "authority_content_sha256": authority["content_sha256"],
        "decisions": [
            {
                "speaker_ref": item["speaker_ref"],
                "decision": "",
                "person_id": None,
                "note": "",
            }
            for item in cases
        ],
    }
    write_immutable_private_json(paths["template"], template)
    paths["html"].write_text(
        build_review_html(
            authority_content_sha256=authority["content_sha256"],
            cases=cases,
            people=bound["people"],
        ),
        encoding="utf-8",
    )
    paths["html"].chmod(0o600)
    receipt = _content(
        {
            "schema_version": RECEIPT_SCHEMA,
            "status": "d2_private_review_ready",
            "preview_content_sha256": preview["content_sha256"],
            "authority_content_sha256": authority["content_sha256"],
            "authority_file_sha256": sha256_file(paths["authority"]),
            "html_file_sha256": sha256_file(paths["html"]),
            "template_file_sha256": sha256_file(paths["template"]),
            "clip_set_sha256": _hash([item["clip_sha256"] for item in cases]),
            "original_recording_filename_set_sha256": preview[
                "original_recording_filename_set_sha256"
            ],
            "recording_count": 6,
            "speaker_slot_count": 18,
            "original_recording_filename_count": 6,
            "human_decision_count": 0,
            "model_predictions_visible": False,
            "fresh_evaluation_allowed": False,
            "mutation_effect_counts": dict(MUTATION_EFFECT_COUNTS),
        }
    )
    write_immutable_private_json(paths["receipt"], receipt)
    return {
        **receipt,
        "private_review_root": str(paths["run"]),
        "private_html_path": str(paths["html"]),
        "decision_template_path": str(paths["template"]),
        "idempotent_replay": False,
    }


def replay_review(*, runtime_root: Path = DEFAULT_RUNTIME_ROOT) -> dict[str, Any]:
    preview = build_review_preview(runtime_root=runtime_root)
    paths = _paths(runtime_root, preview["content_sha256"])
    for path in (paths["authority"], paths["html"], paths["template"], paths["receipt"]):
        require_private_file(path, paths["run"])
    authority = read_private_object(paths["authority"])
    receipt = read_private_object(paths["receipt"])
    _validate_content(authority, "D2 review authority")
    _validate_content(receipt, "D2 review receipt")
    clips = [paths["run"] / item["clip_relative_path"] for item in authority["cases"]]
    for path in clips:
        require_private_file(path, paths["run"])
    current_source = _source_authority(require_clean=False)
    if (
        receipt.get("preview_content_sha256") != preview["content_sha256"]
        or receipt.get("authority_content_sha256") != authority["content_sha256"]
        or receipt.get("authority_file_sha256") != sha256_file(paths["authority"])
        or receipt.get("html_file_sha256") != sha256_file(paths["html"])
        or receipt.get("template_file_sha256") != sha256_file(paths["template"])
        or receipt.get("clip_set_sha256")
        != _hash([sha256_file(path) for path in clips])
        or authority.get("source_authority", {}).get("module_sha256")
        != current_source.get("module_sha256")
        or len(authority.get("cases") or []) != 18
        or authority.get("model_predictions_visible") is not False
        or authority.get("human_decision_count") != 0
        or authority.get("mutation_effect_counts") != MUTATION_EFFECT_COUNTS
    ):
        raise Plan0071D2ReviewError("The D2 review replay drifted.")
    return {
        **receipt,
        "private_review_root": str(paths["run"]),
        "private_html_path": str(paths["html"]),
        "decision_template_path": str(paths["template"]),
        "idempotent_replay": True,
    }


if __name__ == "__main__":
    print(json.dumps(execute_review(), indent=2, sort_keys=True))
