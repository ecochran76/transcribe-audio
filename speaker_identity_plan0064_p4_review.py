"""Plan 0064 P4 private direct-audio gold review surface."""

from __future__ import annotations

import argparse
from array import array
import html
import json
from pathlib import Path
from pathlib import PurePosixPath
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


PREVIEW_SCHEMA = "transcribe-audio.plan0064-p4-review-preview.v2"
AUTHORITY_SCHEMA = "transcribe-audio.plan0064-p4-review-authority.v2"
RECEIPT_SCHEMA = "transcribe-audio.plan0064-p4-review-receipt.v2"
DECISION_SCHEMA = "transcribe-audio.plan0064-p4-human-gold-decisions.v1"
P3_PREVIEW_CONTENT_SHA256 = (
    "2ec73512fc8122efd79201471473b9ac6f5e7f1197f4a5a9c644eebe1537a55b"
)
P3_RESOLUTION_CONTENT_SHA256 = (
    "2f55e7adb9a48e44073e402bd3bc802ddc10c518cdb3d158d00f5a5058492dcb"
)
P3_RECEIPT_CONTENT_SHA256 = (
    "b630d12d6ce21804d8cd0ad4e24ff6f22730ad365c0ea271f9e2db6d661d115e"
)
P2_PREVIEW_CONTENT_SHA256 = (
    "d6014903bf89a4398d3fd392b9feae65d9105c093f21264d954a2649c5253a23"
)
P2_RECEIPT_CONTENT_SHA256 = (
    "50a7f4fd15b8c65c1faf4628309e72796661ac7760651eb7c9666d9117d9bd6b"
)
P2_HYDRATION_BRIDGE_CONTENT_SHA256 = (
    "fc0f3a506492741623516f5aff7d7a5674f797a72a4f1bbb3aac18480cdae222"
)
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
    root = runtime_root.expanduser().absolute()
    p0_root = root / f"p0-{p0_content_sha256[:24]}"
    manifest_path, p0_receipt_path = (
        p0_root / "private-manifest.json",
        p0_root / "receipt.json",
    )
    p3_root = root / f"p3-{P3_PREVIEW_CONTENT_SHA256[:24]}"
    resolution_path, p3_receipt_path = (
        p3_root / "private-resolution.json",
        p3_root / "receipt.json",
    )
    for path, private_root in (
        (manifest_path, root),
        (p0_receipt_path, root),
        (resolution_path, root),
        (p3_receipt_path, root),
    ):
        require_private_file(path, private_root)
    manifest, p0_receipt = _read(manifest_path), _read(p0_receipt_path)
    resolution, p3_receipt = _read(resolution_path), _read(p3_receipt_path)
    if (
        manifest.get("content_sha256") != p0_content_sha256
        or _content_addressed(manifest) != manifest
        or p0_receipt.get("manifest_content_sha256") != p0_content_sha256
        or p0_receipt.get("manifest_file_sha256") != sha256_file(manifest_path)
        or _content_addressed(p0_receipt) != p0_receipt
        or any((p0_receipt.get("action_counts") or {}).values())
        or resolution.get("content_sha256") != P3_RESOLUTION_CONTENT_SHA256
        or resolution.get("preview_content_sha256") != P3_PREVIEW_CONTENT_SHA256
        or _content_addressed(resolution) != resolution
        or any((resolution.get("action_counts") or {}).values())
        or resolution.get("contains_gold") is not False
        or resolution.get("will_apply_speaker_identity") is not False
        or p3_receipt.get("content_sha256") != P3_RECEIPT_CONTENT_SHA256
        or p3_receipt.get("preview_content_sha256") != P3_PREVIEW_CONTENT_SHA256
        or p3_receipt.get("resolution_content_sha256")
        != P3_RESOLUTION_CONTENT_SHA256
        or p3_receipt.get("resolution_file_sha256") != sha256_file(resolution_path)
        or _content_addressed(p3_receipt) != p3_receipt
        or any((p3_receipt.get("action_counts") or {}).values())
    ):
        raise Plan0064P4ReviewError("The frozen P0/P3 review authority drifted.")
    return (
        manifest,
        {
            **p3_receipt,
            "private_resolution_path": str(resolution_path),
            "idempotent_replay": True,
        },
        resolution,
    )


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


def _transcript_hash_allowlist(
    recordings: Sequence[Mapping[str, Any]],
    *,
    p0_content_sha256: str,
    runtime_root: Path,
) -> dict[str, frozenset[str]]:
    root = runtime_root.expanduser().absolute()
    p2_root = root / f"p2-{P2_PREVIEW_CONTENT_SHA256[:24]}"
    bridge_path, receipt_path = (
        p2_root / "identity-hydration-bridge.json",
        p2_root / "receipt.json",
    )
    require_private_file(bridge_path, root)
    require_private_file(receipt_path, root)
    bridge, receipt = _read(bridge_path), _read(receipt_path)
    if (
        bridge.get("content_sha256") != P2_HYDRATION_BRIDGE_CONTENT_SHA256
        or bridge.get("p0_content_sha256") != p0_content_sha256
        or _content_addressed(bridge) != bridge
        or any((bridge.get("action_counts") or {}).values())
        or receipt.get("content_sha256") != P2_RECEIPT_CONTENT_SHA256
        or receipt.get("preview_content_sha256") != P2_PREVIEW_CONTENT_SHA256
        or receipt.get("identity_hydration_bridge_content_sha256")
        != P2_HYDRATION_BRIDGE_CONTENT_SHA256
        or _content_addressed(receipt) != receipt
        or any((receipt.get("action_counts") or {}).values())
    ):
        raise Plan0064P4ReviewError("The frozen P2 hydration authority drifted.")
    changed_by_document_hash = {
        str(row.get("document_id_sha256") or ""): row
        for row in bridge.get("changed_rows") or []
        if isinstance(row, Mapping)
    }
    if len(changed_by_document_hash) != bridge.get("changed_recording_count"):
        raise Plan0064P4ReviewError("The P2 hydration rows are incomplete.")
    allowed: dict[str, frozenset[str]] = {}
    matched_changed: set[str] = set()
    for recording in recordings:
        document_id = str(recording.get("document_id") or "")
        artifact = recording.get("transcript_artifact")
        if not document_id or not isinstance(artifact, Mapping):
            raise Plan0064P4ReviewError("A review recording lacks transcript lineage.")
        old_sha256 = str(artifact.get("sha256") or "")
        values = {old_sha256}
        document_hash = _hash(document_id)
        changed = changed_by_document_hash.get(document_hash)
        if changed is not None:
            if changed.get("old_artifact_sha256") != old_sha256:
                raise Plan0064P4ReviewError("A P2 hydration row has the wrong parent.")
            values.add(str(changed.get("new_artifact_sha256") or ""))
            matched_changed.add(document_hash)
        if any(len(value) != 64 for value in values):
            raise Plan0064P4ReviewError("A transcript hash allowlist is invalid.")
        allowed[document_id] = frozenset(values)
    if matched_changed != set(changed_by_document_hash):
        raise Plan0064P4ReviewError("P2 hydration covers a different recording set.")
    return allowed


def _original_recording_filename(
    recording: Mapping[str, Any], *, allowed_sha256: frozenset[str]
) -> str:
    artifact = recording.get("transcript_artifact")
    if not isinstance(artifact, Mapping):
        raise Plan0064P4ReviewError("A review recording has no transcript authority.")
    path = Path(str(artifact.get("path") or ""))
    try:
        private = (
            path.is_file()
            and not path.is_symlink()
            and sha256_file(path) in allowed_sha256
        )
        transcript = json.loads(path.read_text(encoding="utf-8")) if private else None
    except (OSError, json.JSONDecodeError):
        transcript = None
    if not isinstance(transcript, Mapping):
        raise Plan0064P4ReviewError("A review transcript drifted or is unavailable.")
    source_path = str(transcript.get("source_media_path") or "").strip()
    filename = PurePosixPath(source_path.replace("\\", "/")).name
    if (
        not source_path
        or not filename
        or filename in {".", ".."}
        or len(filename) > 255
        or any(ord(character) < 32 for character in filename)
        or "/" in filename
        or "\\" in filename
    ):
        raise Plan0064P4ReviewError(
            "A review transcript has no safe original recording filename."
        )
    return filename


def _recording_filename_rows(
    recordings: Sequence[Mapping[str, Any]],
    *,
    transcript_hashes: Mapping[str, frozenset[str]],
) -> list[dict[str, str]]:
    rows = [
        {
            "document_id": str(recording.get("document_id") or ""),
            "recording_filename": _original_recording_filename(
                recording,
                allowed_sha256=transcript_hashes.get(
                    str(recording.get("document_id") or ""), frozenset()
                ),
            ),
        }
        for recording in recordings
    ]
    if (
        not rows
        or any(not row["document_id"] for row in rows)
        or len(rows) != len({row["document_id"] for row in rows})
    ):
        raise Plan0064P4ReviewError("Review recording filenames are incomplete.")
    return rows


def _case_recording_filename_rows(
    cases: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    by_document: dict[str, str] = {}
    order: list[str] = []
    for case in cases:
        document_id = str(case.get("document_id") or "")
        filename = str(case.get("recording_filename") or "")
        if not document_id or not filename:
            raise Plan0064P4ReviewError("A review case lacks recording provenance.")
        if document_id not in by_document:
            order.append(document_id)
            by_document[document_id] = filename
        elif by_document[document_id] != filename:
            raise Plan0064P4ReviewError("A recording has conflicting original filenames.")
    return [
        {"document_id": document_id, "recording_filename": by_document[document_id]}
        for document_id in order
    ]


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
    transcript_hashes = _transcript_hash_allowlist(
        selected,
        p0_content_sha256=p0_content_sha256,
        runtime_root=runtime_root,
    )
    filename_rows = _recording_filename_rows(
        selected, transcript_hashes=transcript_hashes
    )
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
            "recording_filename_set_sha256": _hash(filename_rows),
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
                    '<p class="filename"><strong>Original recording:</strong> <code>',
                    html.escape(case["recording_filename"]),
                    '</code></p>',
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
.card h2{{margin:.25rem 0}} .filename{{margin:.5rem 0 1rem;color:var(--muted)}} .filename code{{color:var(--ink);overflow-wrap:anywhere}} audio,select,textarea{{display:block;width:100%;margin:8px 0 16px}} select,textarea,button{{font:inherit;padding:10px;border-radius:8px;border:1px solid #9aa3ad}}
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
    transcript_hashes = _transcript_hash_allowlist(
        selected,
        p0_content_sha256=p0_content_sha256,
        runtime_root=runtime_root,
    )
    filename_by_document = {
        row["document_id"]: row["recording_filename"]
        for row in _recording_filename_rows(
            selected, transcript_hashes=transcript_hashes
        )
    }
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
                    "recording_filename": filename_by_document[recording["document_id"]],
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
            "recording_filename_set_sha256": preview[
                "recording_filename_set_sha256"
            ],
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
            "recording_filename_set_sha256": preview[
                "recording_filename_set_sha256"
            ],
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
    filename_rows = _case_recording_filename_rows(authority["cases"])
    clips = [root / item["clip_relative_path"] for item in authority["cases"]]
    for path in clips:
        require_private_file(path, root)
    if (
        authority.get("content_sha256")
        != _hash({key: value for key, value in authority.items() if key != "content_sha256"})
        or receipt.get("preview_content_sha256") != preview["content_sha256"]
        or receipt.get("authority_content_sha256") != authority["content_sha256"]
        or authority.get("recording_filename_set_sha256")
        != preview["recording_filename_set_sha256"]
        or _hash(filename_rows) != preview["recording_filename_set_sha256"]
        or receipt.get("recording_filename_set_sha256")
        != preview["recording_filename_set_sha256"]
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
