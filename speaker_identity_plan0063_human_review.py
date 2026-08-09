"""Build the private, non-applying Plan 0063 grouping/source review."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from acoustic_audio_derivatives import (
    canonical_artifact_hash,
    ensure_private_tree,
    read_private_object,
    require_private_file,
    sha256_file,
    write_immutable_private_json,
)
import speaker_identity_plan0063_enrollment_feasibility as feasibility
import speaker_identity_plan0063_reconciliation as reconciliation


REVIEW_SCHEMA = "transcribe-audio.plan0063-human-review.v1"
RECEIPT_SCHEMA = "transcribe-audio.plan0063-human-review-receipt.v1"
SUBMISSION_SCHEMA = "transcribe-audio.plan0063-human-review-submission.v1"
RECONCILIATION_SHA256 = reconciliation.RECONCILIATION_SHA256 if hasattr(
    reconciliation, "RECONCILIATION_SHA256"
) else "82a6834165b20e9457536fbbe67e1540a583ee6dd72374296de55e5b6ccf7f05"
FEASIBILITY_SHA256 = (
    "99078e24c28cc94727eda8a05147f7cd533def6069f2dea370978d31376bfb1c"
)
DEFAULT_RUNTIME_ROOT = Path.home() / ".local/state/transcribe-audio/plan-0063"
DEFAULT_P1_ROOT = DEFAULT_RUNTIME_ROOT / "p3-audio-lineage"
DECISION_KEY_RE = re.compile(
    r"^(?:MERGE::person-merge-[a-f0-9]{24}|"
    r"BINDING::voice-person-binding-[a-f0-9]{24}|"
    r"SOURCE::review-window-[a-f0-9]{24})$"
)
NEGATIVE_ACTIONS = dict(reconciliation.NEGATIVE_ACTIONS)


class Plan0063HumanReviewError(ValueError):
    """Raised when the private Plan 0063 review cannot remain exact."""


def _fail(message: str) -> None:
    raise Plan0063HumanReviewError(message)


def _git(repo_root: Path, arguments: Sequence[str]) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        _fail("Repository authority could not be read.")
    return result.stdout.strip()


def _repository_authority() -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    if _git(root, ["status", "--porcelain=v1", "--untracked-files=normal"]):
        _fail("Repository must be clean before the human-review freeze.")
    if _git(
        root, ["rev-list", "--left-right", "--count", "HEAD...@{upstream}"]
    ).split() != ["0", "0"]:
        _fail("Repository must be upstream-even before the human-review freeze.")
    modules = (
        Path(__file__).resolve(),
        root / "speaker_identity_plan0063_reconciliation.py",
        root / "speaker_identity_plan0063_enrollment_feasibility.py",
    )
    return {
        "commit": _git(root, ["rev-parse", "HEAD"]),
        "upstream": _git(root, ["rev-parse", "@{upstream}"]),
        "modules": {path.name: sha256_file(path) for path in modules},
    }


def _person_labels(reconciled: Mapping[str, Any]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for raw in reconciled.get("person_proposals") or []:
        if not isinstance(raw, Mapping):
            _fail("A person proposal is invalid.")
        person_id = str(raw.get("proposed_person_id") or "")
        names = raw.get("member_names")
        if (
            not person_id
            or not isinstance(names, list)
            or len(names) != 1
            or not isinstance(names[0], str)
            or not names[0].strip()
            or person_id in labels
        ):
            _fail("A person proposal label is incomplete or duplicated.")
        labels[person_id] = " ".join(names[0].split())
    if len(labels) != 6:
        _fail("The exact person-proposal denominator drifted.")
    return labels


def _review_source_audio(
    window: Mapping[str, Any], *, p1_root: Path
) -> Path:
    lineage = window.get("lineage")
    if not isinstance(lineage, Mapping):
        _fail("A source window is missing lineage.")
    manifest_path = Path(str(lineage.get("manifest_path") or ""))
    require_private_file(manifest_path, p1_root)
    if (
        sha256_file(manifest_path) != lineage.get("manifest_sha256")
        or lineage.get("validation_status") != "verified_active_metadata_receipt"
    ):
        _fail("A source-window lineage receipt drifted.")
    manifest = read_private_object(manifest_path)
    artifact_path = Path(str(manifest.get("artifact_path") or ""))
    require_private_file(artifact_path, p1_root)
    if (
        manifest.get("run_id") != lineage.get("run_id")
        or sha256_file(artifact_path) != lineage.get("artifact_sha256")
    ):
        _fail("A source-window audio artifact drifted.")
    return artifact_path


def _extract_clip(source: Path, start: float, end: float, target: Path) -> None:
    duration = round(end - start, 3)
    if start < 0 or duration < 3 or duration > 15.001:
        _fail("A review source window is outside the bounded duration.")
    result = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{duration:.3f}",
            "-i",
            str(source),
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
    if result.returncode or not target.is_file():
        _fail("A private review clip could not be extracted.")
    target.chmod(0o600)


def _decision(
    *, key: str, choices: Sequence[str], label: str, details: Mapping[str, Any]
) -> dict[str, Any]:
    if not DECISION_KEY_RE.fullmatch(key) or not choices:
        _fail("A human-review decision definition is invalid.")
    return {
        "decision_key": key,
        "choices": list(choices),
        "selected": None,
        "display_label": label,
        **dict(details),
    }


def build_review_manifest(
    reconciled: Mapping[str, Any],
    source_feasibility: Mapping[str, Any],
    *,
    clip_sha256_by_reference: Mapping[str, str],
    repository_authority: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the exact blank review denominator from frozen P2/P3 evidence."""

    if (
        reconciled.get("content_sha256") != RECONCILIATION_SHA256
        or reconciled.get("status") != "pending_human_grouping_and_binding_review"
        or source_feasibility.get("content_sha256") != FEASIBILITY_SHA256
        or source_feasibility.get("status")
        != "source_feasibility_ready_pending_human_review"
        or source_feasibility.get("reconciliation_content_sha256")
        != RECONCILIATION_SHA256
        or any((reconciled.get("negative_actions") or {}).values())
        or any((source_feasibility.get("negative_actions") or {}).values())
    ):
        _fail("The frozen P2/P3 review inputs drifted.")
    labels = _person_labels(reconciled)
    merges = []
    for raw in reconciled.get("merge_proposals") or []:
        if not isinstance(raw, Mapping) or raw.get("decision") != "pending":
            _fail("A merge proposal is not pending literal review.")
        person_id = str(raw.get("proposed_person_id") or "")
        merge_id = str(raw.get("merge_proposal_id") or "")
        merges.append(
            _decision(
                key=f"MERGE::{merge_id}",
                choices=("accept", "reject"),
                label=labels.get(person_id, ""),
                details={
                    "merge_proposal_id": merge_id,
                    "proposed_person_id": person_id,
                    "basis": raw.get("basis"),
                    "member_slot_ids": list(raw.get("member_slot_ids") or []),
                },
            )
        )
    bindings = []
    for raw in reconciled.get("voice_binding_proposals") or []:
        if not isinstance(raw, Mapping) or raw.get("decision") != "pending":
            _fail("A voice/person binding is not pending literal review.")
        person_id = str(raw.get("proposed_person_id") or "")
        binding_id = str(raw.get("binding_proposal_id") or "")
        bindings.append(
            _decision(
                key=f"BINDING::{binding_id}",
                choices=("same_person", "different_person"),
                label=labels.get(person_id, ""),
                details={
                    "binding_proposal_id": binding_id,
                    "proposed_person_id": person_id,
                    "acoustic_subject_id": raw.get("acoustic_subject_id"),
                    "member_slot_ids": [raw.get("slot_id")],
                },
            )
        )
    sources = []
    seen_references: set[str] = set()
    for raw in source_feasibility.get("person_source_proposals") or []:
        if (
            not isinstance(raw, Mapping)
            or raw.get("status") != "source_feasible_pending_human_review"
            or raw.get("enrollment_authorized") is not False
        ):
            _fail("A source proposal is not pending literal review.")
        person_id = str(raw.get("proposed_person_id") or "")
        windows = []
        for window in raw.get("source_windows") or []:
            if not isinstance(window, Mapping):
                _fail("A source window is invalid.")
            reference_id = str(window.get("reference_id") or "")
            if (
                reference_id in seen_references
                or window.get("future_holdout_excluded") is not True
                or window.get("data_split") != "development_training_candidate"
                or reference_id not in clip_sha256_by_reference
            ):
                _fail("A source window is duplicated, unbound, or holdout-unsafe.")
            seen_references.add(reference_id)
            windows.append(
                _decision(
                    key=f"SOURCE::{reference_id}",
                    choices=("include", "exclude"),
                    label=labels.get(person_id, ""),
                    details={
                        "reference_id": reference_id,
                        "proposed_person_id": person_id,
                        "slot_id": window.get("slot_id"),
                        "speaker_label_id": window.get("speaker_label_id"),
                        "start_seconds": window.get("start_seconds"),
                        "end_seconds": window.get("end_seconds"),
                        "source_sha256": window.get("source_sha256"),
                        "clip_url": f"clips/{reference_id}.wav",
                        "clip_sha256": clip_sha256_by_reference[reference_id],
                    },
                )
            )
        sources.append(
            {
                "proposed_person_id": person_id,
                "display_label": labels.get(person_id, ""),
                "member_slot_ids": list(raw.get("member_slot_ids") or []),
                "device_metadata_status": raw.get("device_metadata_status"),
                "windows": windows,
            }
        )
    if (len(merges), len(bindings), len(sources), len(seen_references)) != (
        3,
        1,
        5,
        26,
    ):
        _fail("The combined human-review denominator drifted.")
    clinical_slots = {
        "47ea79857aa1ac2d1d79::SPEAKER_2",
        "47ea79857aa1ac2d1d79::SPEAKER_3",
    }
    calendar_person = next(
        (
            raw
            for raw in reconciled.get("person_proposals") or []
            if isinstance(raw, Mapping)
            and set(raw.get("member_slot_ids") or []) == clinical_slots
        ),
        None,
    )
    if not isinstance(calendar_person, Mapping):
        _fail("The reviewed calendar-title candidate is missing.")
    calendar_person_id = str(calendar_person.get("proposed_person_id") or "")
    core = {
        "schema_version": REVIEW_SCHEMA,
        "status": "blank_human_review_pending",
        "reconciliation_content_sha256": RECONCILIATION_SHA256,
        "feasibility_content_sha256": FEASIBILITY_SHA256,
        "repository_authority": dict(repository_authority),
        "calendar_candidate_observation": {
            "proposed_person_id": calendar_person_id,
            "display_label": labels[calendar_person_id],
            "authority": "operator_observed_calendar_title",
            "stored_calendar_source_status": "correct_source_not_captured",
            "candidate_only": True,
            "speaker_assignment_proven": False,
        },
        "merge_reviews": merges,
        "binding_reviews": bindings,
        "source_reviews": sources,
        "decision_count": 30,
        "live_mutation_count": 0,
        "negative_actions": NEGATIVE_ACTIONS,
    }
    return {**core, "content_sha256": canonical_artifact_hash(core)}


def _radio_group(decision: Mapping[str, Any], legend: str) -> str:
    key = html.escape(str(decision["decision_key"]), quote=True)
    choices = []
    for choice in decision["choices"]:
        value = html.escape(str(choice), quote=True)
        display = str(choice).replace("_", " ").capitalize()
        choices.append(
            f'<label><input type="radio" name="{key}" value="{value}"> '
            f"{html.escape(display)}</label>"
        )
    return (
        f'<fieldset class="decision" data-key="{key}"><legend>{html.escape(legend)}</legend>'
        + "".join(choices)
        + "</fieldset>"
    )


def render_review_html(manifest: Mapping[str, Any]) -> str:
    """Render a standalone no-submit page with strict local export."""

    decisions = [
        *manifest.get("merge_reviews", []),
        *manifest.get("binding_reviews", []),
        *[
            window
            for person in manifest.get("source_reviews", [])
            for window in person.get("windows", [])
        ],
    ]
    if len(decisions) != manifest.get("decision_count"):
        _fail("The review HTML decision denominator drifted.")
    merge_cards = []
    for item in manifest["merge_reviews"]:
        slots = " and ".join(html.escape(value) for value in item["member_slot_ids"])
        merge_cards.append(
            '<article class="card"><h3>'
            + html.escape(item["display_label"])
            + "</h3><p>Should these reviewed slots be one person?</p><code>"
            + slots
            + "</code>"
            + _radio_group(item, "Grouping decision")
            + "</article>"
        )
    binding_cards = []
    for item in manifest["binding_reviews"]:
        binding_cards.append(
            '<article class="card"><h3>'
            + html.escape(item["display_label"])
            + "</h3><p>The enrolled voice subject and contextual person were both suggested. Are they the same person?</p>"
            + _radio_group(item, "Voice/context binding")
            + "</article>"
        )
    source_cards = []
    for person in manifest["source_reviews"]:
        rows = []
        for index, window in enumerate(person["windows"], start=1):
            rows.append(
                '<div class="clip"><h4>Clip '
                + str(index)
                + "</h4><audio controls preload=\"none\" src=\""
                + html.escape(window["clip_url"], quote=True)
                + '\"></audio><p class="hint">Include only if this is the named person and is usable as voice training data.</p>'
                + _radio_group(window, "Source decision")
                + "</div>"
            )
        source_cards.append(
            '<article class="card source"><h3>'
            + html.escape(person["display_label"])
            + "</h3><p>Device metadata is unverified. These clips are development/training candidates and excluded from future holdouts.</p>"
            + "".join(rows)
            + "</article>"
        )
    calendar = manifest["calendar_candidate_observation"]
    headers = [
        f"PLAN0063_SCHEMA={SUBMISSION_SCHEMA}",
        f"PLAN0063_P2_CONTENT_SHA256={manifest['reconciliation_content_sha256']}",
        f"PLAN0063_P3_CONTENT_SHA256={manifest['feasibility_content_sha256']}",
        f"PLAN0063_P4_CONTENT_SHA256={manifest['content_sha256']}",
    ]
    script_data = json.dumps(
        {
            "decisionKeys": [item["decision_key"] for item in decisions],
            "allowed": {
                item["decision_key"]: item["choices"] for item in decisions
            },
            "headers": headers,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).replace("<", "\\u003c")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Plan 0063 speaker grouping and source review</title>
<style>
:root{{--bg:#f5f2ea;--ink:#22231f;--card:#fffdf8;--accent:#315e52;--line:#d8d2c4;--warn:#8b541d}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.45 system-ui,sans-serif}}main{{max-width:980px;margin:auto;padding:24px}}h1{{font-size:clamp(1.7rem,4vw,2.6rem);line-height:1.08}}h2{{margin-top:40px}}.notice,.card,.export{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px;margin:14px 0;box-shadow:0 5px 18px #312b1f12}}.notice{{border-left:6px solid var(--warn)}}code{{display:block;overflow-wrap:anywhere;white-space:normal}}.decision{{border:0;padding:8px 0 0;margin:10px 0}}.decision label{{display:inline-flex;gap:7px;align-items:center;margin:6px 16px 6px 0;padding:9px 12px;border:1px solid var(--line);border-radius:999px;cursor:pointer}}.clip{{border-top:1px solid var(--line);padding:14px 0}}audio{{width:100%;max-width:560px}}.hint{{color:#555;font-size:.92rem}}button{{border:0;border-radius:10px;background:var(--accent);color:white;padding:12px 16px;font-weight:700;margin:4px 8px 4px 0;cursor:pointer}}button.secondary{{background:#555}}textarea{{width:100%;min-height:260px;margin-top:12px;padding:12px;font:13px/1.35 ui-monospace,monospace}}#status{{min-height:1.5em;font-weight:650}}.missing{{outline:3px solid #bd2d2d55;border-radius:8px}}
</style></head><body><main>
<h1>Speaker grouping and voice-source review</h1>
<p>This review has 3 grouping decisions, 1 enrolled-voice/context binding, and 26 exact source clips. It starts blank and cannot apply changes.</p>
<section class="notice"><h2>Calendar-source gap</h2><p>The reviewed calendar-title candidate is <strong>{html.escape(calendar['display_label'])}</strong>. The corrected source calendar was not captured in the stored event evidence, so this remains an operator-observed candidate—not proof that the person spoke. The implementation can cite the title once that calendar source is available.</p></section>
<form id="review-form" action="" method="get" onsubmit="return false">
<h2>1. Person grouping</h2>{''.join(merge_cards)}
<h2>2. Existing voice and contextual person</h2>{''.join(binding_cards)}
<h2>3. New voice enrollment sources</h2>{''.join(source_cards)}
<section class="export"><h2>Answer block</h2><p>Complete every choice, then build and copy the exact hash-bound block. No network request is made.</p><div><button type="button" id="build">Build answer block</button><button type="button" class="secondary" id="copy">Copy answer block</button></div><p id="status" role="status" aria-live="polite"></p><textarea id="answer" readonly spellcheck="false" aria-label="Answer block"></textarea></section>
</form></main><script>
const review={script_data};
const form=document.querySelector('#review-form');
const answer=document.querySelector('#answer');
const status=document.querySelector('#status');
function build(){{
  document.querySelectorAll('.missing').forEach(el=>el.classList.remove('missing'));
  const rows=[...review.headers];
  let firstMissing=null;
  for(const key of review.decisionKeys){{
    const selected=form.querySelector(`input[name="${{CSS.escape(key)}}"]:checked`);
    if(!selected || !review.allowed[key].includes(selected.value)){{
      const field=form.querySelector(`fieldset[data-key="${{CSS.escape(key)}}"]`);
      if(field) field.classList.add('missing');
      firstMissing ||= field;
      continue;
    }}
    rows.push(`${{key}}=${{selected.value}}`);
  }}
  if(firstMissing){{answer.value='';status.textContent='Complete all 30 decisions before exporting.';firstMissing.scrollIntoView({{behavior:'smooth',block:'center'}});return false;}}
  answer.value=rows.join('\\n');status.textContent='Answer block built. Review it, then copy.';return true;
}}
async function copyBlock(){{
  if(!build()) return;
  answer.focus();answer.select();
  try{{await navigator.clipboard.writeText(answer.value);status.textContent='Answer block copied.';}}
  catch(error){{const ok=document.execCommand('copy');status.textContent=ok?'Answer block copied with browser fallback.':'Copy was blocked. The complete block is selected for manual copy.';}}
}}
document.querySelector('#build').addEventListener('click',build);
document.querySelector('#copy').addEventListener('click',copyBlock);
</script></body></html>"""


def parse_review_submission(
    text: str, review_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate one complete literal answer block without applying it."""

    pairs: dict[str, str] = {}
    for raw in text.splitlines():
        if not raw.strip() or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        if key in pairs:
            _fail("The answer block contains a duplicate key.")
        pairs[key] = value
    expected_headers = {
        "PLAN0063_SCHEMA": SUBMISSION_SCHEMA,
        "PLAN0063_P2_CONTENT_SHA256": review_manifest.get(
            "reconciliation_content_sha256"
        ),
        "PLAN0063_P3_CONTENT_SHA256": review_manifest.get(
            "feasibility_content_sha256"
        ),
        "PLAN0063_P4_CONTENT_SHA256": review_manifest.get("content_sha256"),
    }
    for key, value in expected_headers.items():
        if pairs.pop(key, None) != value:
            _fail("The answer block header does not match this review.")
    decisions = [
        *review_manifest.get("merge_reviews", []),
        *review_manifest.get("binding_reviews", []),
        *[
            window
            for person in review_manifest.get("source_reviews", [])
            for window in person.get("windows", [])
        ],
    ]
    allowed = {item["decision_key"]: set(item["choices"]) for item in decisions}
    if set(pairs) != set(allowed) or any(
        value not in allowed[key] for key, value in pairs.items()
    ):
        _fail("Every exact review decision requires one allowlisted value.")
    core = {
        "schema_version": SUBMISSION_SCHEMA,
        "review_content_sha256": review_manifest["content_sha256"],
        "decisions": [
            {"decision_key": key, "decision": pairs[key]}
            for key in sorted(pairs)
        ],
        "live_mutation_count": 0,
    }
    return {**core, "content_sha256": canonical_artifact_hash(core)}


def _paths(runtime_root: Path, content_sha256: str) -> dict[str, Path]:
    root = runtime_root.expanduser().absolute()
    run = root / f"p4-human-review-{content_sha256[:24]}"
    return {
        "root": root,
        "run": run,
        "manifest": run / "private-manifest.json",
        "receipt": run / "receipt.json",
        "html": run / "review.html",
        "clips": run / "clips",
    }


def replay_review(
    *, content_sha256: str, runtime_root: Path = DEFAULT_RUNTIME_ROOT
) -> dict[str, Any]:
    paths = _paths(runtime_root, content_sha256)
    for key in ("manifest", "receipt", "html"):
        require_private_file(paths[key], paths["root"])
    manifest = read_private_object(paths["manifest"])
    receipt = read_private_object(paths["receipt"])
    core = {key: value for key, value in manifest.items() if key != "content_sha256"}
    windows = [
        window
        for person in manifest.get("source_reviews", [])
        for window in person.get("windows", [])
    ]
    for window in windows:
        clip = paths["run"] / str(window.get("clip_url") or "")
        require_private_file(clip, paths["run"])
        if sha256_file(clip) != window.get("clip_sha256"):
            _fail("A private review clip drifted.")
    if (
        manifest.get("schema_version") != REVIEW_SCHEMA
        or manifest.get("content_sha256") != content_sha256
        or canonical_artifact_hash(core) != content_sha256
        or manifest.get("decision_count") != 30
        or any((manifest.get("negative_actions") or {}).values())
        or receipt.get("schema_version") != RECEIPT_SCHEMA
        or receipt.get("content_sha256") != content_sha256
        or receipt.get("manifest_sha256") != sha256_file(paths["manifest"])
        or receipt.get("review_html_sha256") != sha256_file(paths["html"])
        or receipt.get("clip_count") != len(windows)
        or receipt.get("live_mutation_count") != 0
    ):
        _fail("The frozen Plan 0063 human review drifted.")
    return {
        **receipt,
        "manifest_path": str(paths["manifest"]),
        "receipt_path": str(paths["receipt"]),
        "review_directory": str(paths["run"]),
        "review_html_path": str(paths["html"]),
        "idempotent_replay": True,
    }


def freeze_exact_review(
    *,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    p1_root: Path = DEFAULT_P1_ROOT,
) -> dict[str, Any]:
    """Extract exact clips and freeze the blank, non-applying P4 review."""

    p2 = reconciliation.replay_reconciliation(
        content_sha256=RECONCILIATION_SHA256, runtime_root=runtime_root
    )
    p3 = feasibility.replay_feasibility(
        content_sha256=FEASIBILITY_SHA256, runtime_root=runtime_root
    )
    reconciled = read_private_object(Path(p2["manifest_path"]))
    source_feasibility = read_private_object(Path(p3["manifest_path"]))
    ensure_private_tree(runtime_root, runtime_root)
    stage = Path(tempfile.mkdtemp(prefix=".p4-human-review-stage-", dir=runtime_root))
    stage.chmod(0o700)
    clips = stage / "clips"
    clips.mkdir(mode=0o700)
    try:
        clip_hashes: dict[str, str] = {}
        for person in source_feasibility.get("person_source_proposals") or []:
            for window in person.get("source_windows") or []:
                reference_id = str(window.get("reference_id") or "")
                if reference_id in clip_hashes:
                    _fail("A review source reference is duplicated.")
                source = _review_source_audio(window, p1_root=p1_root)
                target = clips / f"{reference_id}.wav"
                _extract_clip(
                    source,
                    float(window.get("start_seconds")),
                    float(window.get("end_seconds")),
                    target,
                )
                clip_hashes[reference_id] = sha256_file(target)
        manifest = build_review_manifest(
            reconciled,
            source_feasibility,
            clip_sha256_by_reference=clip_hashes,
            repository_authority=_repository_authority(),
        )
        paths = _paths(runtime_root, manifest["content_sha256"])
        if paths["run"].exists():
            shutil.rmtree(stage)
            return replay_review(
                content_sha256=manifest["content_sha256"],
                runtime_root=runtime_root,
            )
        html_body = render_review_html(manifest)
        (stage / "review.html").write_text(html_body, encoding="utf-8")
        (stage / "review.html").chmod(0o600)
        write_immutable_private_json(stage / "private-manifest.json", manifest)
        receipt = {
            "schema_version": RECEIPT_SCHEMA,
            "status": "blank_human_review_frozen_pending_operator",
            "content_sha256": manifest["content_sha256"],
            "manifest_sha256": sha256_file(stage / "private-manifest.json"),
            "review_html_sha256": sha256_file(stage / "review.html"),
            "clip_count": len(clip_hashes),
            "decision_count": manifest["decision_count"],
            "live_mutation_count": 0,
            "negative_actions_preserved": True,
        }
        write_immutable_private_json(stage / "receipt.json", receipt)
        os.replace(stage, paths["run"])
        return {
            **receipt,
            "manifest_path": str(paths["manifest"]),
            "receipt_path": str(paths["receipt"]),
            "review_directory": str(paths["run"]),
            "review_html_path": str(paths["html"]),
            "idempotent_replay": False,
        }
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", metavar="CONTENT_SHA256")
    args = parser.parse_args()
    result = (
        replay_review(content_sha256=args.replay)
        if args.replay
        else freeze_exact_review()
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
