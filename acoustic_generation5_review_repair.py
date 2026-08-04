"""Build a self-contained repaired Plan 0055 review page."""

from __future__ import annotations

import base64
import hashlib
import html
import json
import os
from pathlib import Path
from typing import Any, Mapping

import acoustic_generation5_source_review as s1
from acoustic_audio_derivatives import require_private_file, sha256_file


SOURCE_PREVIEW_SHA256 = "5a3f9fc9848a5e0b669bc37796e5a55b4f9dcd7bf0f55609aefa886e4caabcf9"
DEFAULT_RUNTIME_ROOT = Path("~/.local/state/transcribe-audio/plan-0055/s1/review-repair")


class Generation5ReviewRepairError(ValueError):
    """Raised when the repaired review surface cannot remain exact."""


def _canonical_hash(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(body.encode()).hexdigest()


def _validated_source() -> tuple[dict[str, Any], Path]:
    replay = s1.replay_generation5_source_review(SOURCE_PREVIEW_SHA256)
    paths = s1._paths(s1.DEFAULT_RUNTIME_ROOT, SOURCE_PREVIEW_SHA256)
    if replay.get("idempotent_replay") is not True or replay.get("clip_count") != 40:
        raise Generation5ReviewRepairError("The original S1 review authority drifted.")
    require_private_file(paths["manifest"], paths["root"])
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    preview = manifest.get("preview")
    if not isinstance(preview, dict) or preview.get("content_sha256") != SOURCE_PREVIEW_SHA256:
        raise Generation5ReviewRepairError("The original S1 preview is unavailable.")
    return preview, paths["clips"]


def render_standalone_review(
    preview: Mapping[str, Any], clips_root: Path, *, answers: Mapping[str, str],
) -> str:
    cards = preview.get("private_evidence", {}).get("cards")
    if not isinstance(cards, list) or len(cards) != 40:
        raise Generation5ReviewRepairError("Exactly forty review cards are required.")
    expected_refs = {str(card.get("speaker_ref") or "") for card in cards}
    if set(answers) - expected_refs:
        raise Generation5ReviewRepairError("A prefilled answer reference is unknown.")
    sections = []
    for card in cards:
        clip = clips_root / f"{s1._slug(card)}.wav"
        if not clip.is_file() or clip.is_symlink() or clip.stat().st_size <= 44:
            raise Generation5ReviewRepairError("A review clip is unavailable.")
        encoded = base64.b64encode(clip.read_bytes()).decode("ascii")
        snippets = "".join(
            f"<li>{html.escape(str(item['text']))}</li>"
            for item in card["clip"]["snippets"]
        )
        reference = str(card["speaker_ref"])
        value = str(answers.get(reference, ""))
        sections.append(
            f'<section class="card"><h2>{html.escape(reference)}</h2>'
            f'<audio controls preload="metadata" src="data:audio/wav;base64,{encoded}"></audio>'
            f'<details open><summary>Transcript clues</summary><ul>{snippets}</ul></details>'
            f'<label>Identity or stable alias<input data-answer="1" '
            f'data-ref="{html.escape(reference, quote=True)}" '
            f'value="{html.escape(value, quote=True)}"></label>'
            '<p class="hint">Reuse the same identity for the same person. '
            'Use UNKNOWN only if you cannot tell.</p></section>'
        )
    return f'''<!doctype html><html><head><meta charset="utf-8"><title>Generation-5 private speaker review — repaired</title>
<style>body{{font:16px system-ui;max-width:920px;margin:2rem auto;padding:0 1rem;background:#f5f7fa;color:#18202a}}.notice{{background:#fff4cc;border:1px solid #d8b94d;padding:1rem;border-radius:10px}}.card{{background:#fff;padding:1rem 1.2rem;margin:1rem 0;border-radius:12px;box-shadow:0 1px 5px #0002}}audio{{width:100%}}input,textarea{{box-sizing:border-box;display:block;width:100%;padding:.65rem;margin:.5rem 0}}textarea{{min-height:18rem}}button{{font-size:1rem;padding:.75rem 1rem}}.hint{{color:#59636e;font-size:.9rem}}details li{{margin:.5rem 0}}</style></head><body>
<h1>Private Generation-5 speaker-label review — repaired</h1><p class="notice">This replacement embeds all 40 recordings directly in this page. Your previously supplied identities are prefilled. Fill every remaining blank, then click <strong>Prepare answers</strong>. If clipboard permission is denied, the complete block still appears and is selected below.</p>
<p>Required A is the Zoom recording. Required B is the Agritalk recording. Enrolled people to look for: Chris Williams and Eric Cochran.</p>
<button id="prepare" type="button">Prepare answers</button><span id="status" role="status"></span><textarea id="answers" aria-label="Copyable answer block" placeholder="The complete answer block will appear here."></textarea>{''.join(sections)}
<script>
function prepareAnswers() {{
  const lines = Array.from(document.querySelectorAll('[data-answer]')).map(
    (item) => `${{item.dataset.ref}} = ${{item.value.trim() || 'UNANSWERED'}}`
  );
  const text = lines.join('\\n');
  const box = document.getElementById('answers');
  const status = document.getElementById('status');
  box.value = text;
  box.focus();
  box.select();
  if (navigator.clipboard && navigator.clipboard.writeText) {{
    navigator.clipboard.writeText(text).then(
      () => {{ status.textContent = ' Copied — paste into chat.'; }},
      () => {{ status.textContent = ' Copy the selected answer block below.'; }}
    );
  }} else {{
    status.textContent = ' Copy the selected answer block below.';
  }}
}}
document.getElementById('prepare').addEventListener('click', prepareAnswers);
</script></body></html>'''


def build_repaired_review(answers: Mapping[str, str]) -> dict[str, Any]:
    preview, clips_root = _validated_source()
    normalized_answers = {
        str(reference): " ".join(str(value).split())
        for reference, value in answers.items()
        if str(value).strip()
    }
    page = render_standalone_review(preview, clips_root, answers=normalized_answers)
    answer_set_sha256 = _canonical_hash(normalized_answers)
    content_sha256 = hashlib.sha256(page.encode("utf-8")).hexdigest()
    root = DEFAULT_RUNTIME_ROOT.expanduser().absolute()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    root.chmod(0o700)
    path = root / f"generation5-review-repaired-{content_sha256[:24]}.html"
    if path.exists():
        require_private_file(path, root)
        if sha256_file(path) != content_sha256:
            raise Generation5ReviewRepairError("The repaired page changed in place.")
    else:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(page)
            stream.flush()
            os.fsync(stream.fileno())
    return {
        "path": str(path),
        "content_sha256": content_sha256,
        "source_preview_sha256": SOURCE_PREVIEW_SHA256,
        "answer_set_sha256": answer_set_sha256,
        "prefilled_answer_count": len(normalized_answers),
        "card_count": 40,
        "embedded_audio_count": page.count("data:audio/wav;base64,"),
        "mode": "0600",
    }
