#!/usr/bin/env python3
"""
Browser-smoke the M1 dogfoodable conversation review loop.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import urlencode
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.smoke_app_replay_manifest import DEFAULT_BASE_URL
from scripts.smoke_app_replay_manifest_ui import DEFAULT_REPORT_DIR


CONVERSATION_REVIEW_LOOP_SMOKE_JSON_STDOUT_PREFIX = "CONVERSATION_REVIEW_LOOP_SMOKE_JSON="


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Browser-smoke the conversation review loop workspace.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Transcript console base URL.")
    parser.add_argument("--session", default="transcript-conversation-review-loop-smoke", help="agent-browser session name.")
    parser.add_argument("--profile", type=Path, default=None, help="Optional browser profile path.")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR, help="Directory for JSON/screenshot artifacts.")
    parser.add_argument("--query", default="Tempo", help="Library search query for a contextual conversation.")
    parser.add_argument("--kind", default="contextual_readout", help="Library kind filter.")
    parser.add_argument("--viewport", default="1360x860", help="Viewport size as WIDTHxHEIGHT.")
    return parser.parse_args(argv)


def resolve_agent_browser_bin() -> str:
    configured = os.environ.get("AGENT_BROWSER_BIN", "").strip()
    candidates = [
        configured,
        shutil.which("agent-browser") or "",
        str(Path("~/.local/bin/agent-browser").expanduser()),
        str(Path("~/.local/share/pnpm/agent-browser").expanduser()),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    raise FileNotFoundError("agent-browser is not on PATH; set AGENT_BROWSER_BIN or install ~/.local/bin/agent-browser.")


def parse_viewport(value: str) -> tuple[int, int]:
    width, height = value.lower().split("x", 1)
    return int(width), int(height)


def run_agent_browser(agent_browser_bin: str, session: str, profile: Path, command: list[str], *, timeout: int = 45) -> str:
    proc = subprocess.run(
        [agent_browser_bin, "--profile", str(profile), "--session", session, *command],
        check=True,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    return proc.stdout.strip()


def eval_js(agent_browser_bin: str, session: str, profile: Path, script: str) -> Any:
    output = run_agent_browser(agent_browser_bin, session, profile, ["eval", script])
    return json.loads(output)


def click_tab(agent_browser_bin: str, session: str, profile: Path, label: str) -> None:
    eval_js(
        agent_browser_bin,
        session,
        profile,
        f"""
(() => {{
  const button = [...document.querySelectorAll('.workflow-view-tabs button')]
    .find((item) => item.textContent.trim() === {json.dumps(label)});
  if (!button) throw new Error(`Missing workflow tab: {label}`);
  button.click();
  return true;
}})()
""".strip(),
    )
    run_agent_browser(agent_browser_bin, session, profile, ["wait", "350"])


def first_conversation_id(base_url: str, *, kind: str, query: str) -> str:
    params = urlencode({"kind": kind, "query": query, "limit": "1"})
    with urlopen(f"{base_url.rstrip('/')}/api/conversations?{params}", timeout=20) as response:
        payload = json.loads(response.read())
    items = payload.get("items") if isinstance(payload, dict) else []
    first = items[0] if items else {}
    representative = first.get("representative") if isinstance(first.get("representative"), dict) else {}
    return str(representative.get("id") or "")


def conversation_detail(base_url: str, document_id: str) -> dict[str, Any]:
    with urlopen(f"{base_url.rstrip('/')}/api/conversations/{document_id}", timeout=30) as response:
        payload = json.loads(response.read())
    return payload if isinstance(payload, dict) else {}


def collect_checks(agent_browser_bin: str, session: str, profile: Path) -> dict[str, Any]:
    return eval_js(
        agent_browser_bin,
        session,
        profile,
        """
(() => {
  const text = document.body.innerText;
  const lowerText = text.toLowerCase();
  const activeTab = [...document.querySelectorAll('.workflow-view-tabs button')]
    .find((button) => button.getAttribute('aria-pressed') === 'true')?.textContent.trim() || '';
  const queuePreviewButton = [...document.querySelectorAll('button')]
    .find((button) => button.textContent.trim() === 'Queue preview review');
  const summaryPrimaryActions = [...document.querySelectorAll('.workflow-prep-card .primary-workflow-action')];
  const inlineSummaryActionRow = document.querySelector('.workflow-action-panel > .workflow-action-row');
  const advancedSummaryControls = document.querySelector('.workflow-secondary-actions');
  return {
    activeTab,
    hasConversationWorkspace: lowerText.includes('conversation workspace'),
    isConversationLoading: lowerText.includes('loading conversation workspace'),
    hasSourceAudio: Boolean(document.querySelector('.conversation-rail audio[src*="/api/blobs/"]')),
    hasTranscriptTurns: document.querySelectorAll('.transcript-turn').length > 0,
    hasSummaryReady: text.includes('Summary ready') || text.includes('First-pass summary'),
    hasSummaryPrepCard: Boolean(document.querySelector('.workflow-prep-card')),
    summaryPrimaryActionCount: summaryPrimaryActions.length,
    summaryPrimaryActionText: summaryPrimaryActions.map((button) => button.textContent.trim()).join(' | '),
    hasAdvancedSummaryControls: Boolean(advancedSummaryControls),
    advancedSummaryControlsText: advancedSummaryControls?.textContent || '',
    hasInlineSummaryButtonCluster: Boolean(inlineSummaryActionRow),
    hasSpeakerReview: text.includes('pending assignments') || text.includes('Speaker assignment'),
    hasManualContactInput: Boolean(document.querySelector('input[aria-label^="Manual contact for"]')),
    hasManualContactButton: text.includes('Confirm typed'),
    hasCalendarEvidence: text.includes('Calendar evidence'),
    hasContextWorkbench: text.includes('Included provenance') && text.includes('Excluded provenance'),
    hasParticipantIdentity: lowerText.includes('participant identity'),
    hasIdentitySourceProfileChip: [...document.querySelectorAll('.chip-cloud span')]
      .some((item) => /^(gws|odollo):/i.test(item.textContent.trim())),
    hasFinalPreview: lowerText.includes('deposition and memory preview'),
    hasFinalPreviewBlocked: lowerText.includes('identity or context review is still required')
      || lowerText.includes('preview is blocked until identity'),
    hasMemoryCandidates: lowerText.includes('memory candidate'),
    hasQueuePreviewButton: text.includes('Queue preview review'),
    hasQueuePreviewButtonDisabled: Boolean(queuePreviewButton?.disabled),
    urlConversation: new URL(window.location.href).searchParams.get('conversation'),
    urlWorkflow: new URL(window.location.href).searchParams.get('workflow')
  };
})()
""".strip(),
    )


def wait_for_workspace_ready(
    agent_browser_bin: str,
    session: str,
    profile: Path,
    *,
    timeout: float = 60.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_checks: dict[str, Any] = {}
    while time.monotonic() < deadline:
        last_checks = collect_checks(agent_browser_bin, session, profile)
        if last_checks.get("hasConversationWorkspace") and not last_checks.get("isConversationLoading"):
            return last_checks
        run_agent_browser(agent_browser_bin, session, profile, ["wait", "750"])
    raise TimeoutError("Conversation workspace did not finish loading before the smoke timeout.")


def run_smoke(
    *,
    base_url: str,
    session: str,
    profile: Path,
    report_dir: Path,
    query: str,
    kind: str,
    viewport: tuple[int, int],
) -> dict[str, Any]:
    agent_browser_bin = resolve_agent_browser_bin()
    base_url = base_url.rstrip("/")
    report_dir = report_dir.expanduser()
    report_dir.mkdir(parents=True, exist_ok=True)
    profile = profile.expanduser()
    profile.mkdir(parents=True, exist_ok=True)
    width, height = viewport
    selected_id = first_conversation_id(base_url, kind=kind, query=query)
    detail = conversation_detail(base_url, selected_id) if selected_id else {}
    identity_review = detail.get("identity_review") if isinstance(detail.get("identity_review"), dict) else {}
    identity_bundle = identity_review.get("identity_bundle") if isinstance(identity_review.get("identity_bundle"), dict) else {}
    context_workbench = detail.get("context_workbench") if isinstance(detail.get("context_workbench"), dict) else {}
    context_identity_bundle = (
        context_workbench.get("participant_identity_bundle")
        if isinstance(context_workbench.get("participant_identity_bundle"), dict)
        else {}
    )
    final_preview = detail.get("final_preview") if isinstance(detail.get("final_preview"), dict) else {}
    final_identity_warnings = final_preview.get("identity_context_warnings")
    if not isinstance(final_identity_warnings, list):
        final_identity_warnings = []
    api_final_blocked = final_preview.get("status") == "blocked_identity_or_context_review" or bool(final_identity_warnings)
    api_speaker_count = len(identity_bundle.get("speaker_labels") or [])
    api_source_profile_count = len(identity_bundle.get("source_profiles") or [])
    api_calendar_attendee_count = len(identity_bundle.get("calendar_attendees") or [])
    params = urlencode(
        {
            "kind": kind,
            "q": query,
            "conversation": "1",
            "workflow": "transcript",
            "selected": selected_id,
        }
    )
    deep_link_url = f"{base_url}/?{params}"

    run_agent_browser(agent_browser_bin, session, profile, ["set", "viewport", str(width), str(height)])
    run_agent_browser(agent_browser_bin, session, profile, ["open", deep_link_url])
    run_agent_browser(agent_browser_bin, session, profile, ["wait", "2200"])
    transcript_checks = wait_for_workspace_ready(agent_browser_bin, session, profile)
    checks: dict[str, Any] = {}
    checks.update(
        {
            "api_hasParticipantIdentityBundle": identity_bundle.get("schema_version")
            == "transcribe-audio.participant-identity-bundle.v1",
            "api_contextHasParticipantIdentityBundle": context_identity_bundle.get("schema_version")
            == "transcribe-audio.participant-identity-bundle.v1",
            "api_identitySpeakerCount": api_speaker_count,
            "api_identityCandidateCount": len(identity_bundle.get("contact_candidates") or []),
            "api_identityPendingCount": int(identity_review.get("pending_count") or 0),
            "api_identitySourceProfileCount": api_source_profile_count,
            "api_identityCalendarAttendeeCount": api_calendar_attendee_count,
            "api_finalBlockedByIdentityOrContext": api_final_blocked,
            "api_finalIdentityContextWarningCount": len(final_identity_warnings),
        }
    )
    checks.update({f"transcript_{key}": value for key, value in transcript_checks.items()})
    for label, prefix in [
        ("First-pass summary", "summary"),
        ("Speakers", "speakers"),
        ("Context workbench", "context"),
        ("Final readout", "final"),
    ]:
        click_tab(agent_browser_bin, session, profile, label)
        checks.update({f"{prefix}_{key}": value for key, value in collect_checks(agent_browser_bin, session, profile).items()})

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    screenshot_path = report_dir / f"{stamp}-conversation-review-loop-smoke.png"
    run_agent_browser(agent_browser_bin, session, profile, ["screenshot", str(screenshot_path)], timeout=45)
    expected = {
        "transcript_hasConversationWorkspace": True,
        "transcript_hasSourceAudio": True,
        "transcript_hasTranscriptTurns": True,
        "summary_hasSummaryReady": True,
        "summary_hasSummaryPrepCard": True,
        "summary_summaryPrimaryActionCount": 1,
        "summary_hasAdvancedSummaryControls": True,
        "summary_hasInlineSummaryButtonCluster": False,
        "speakers_hasSpeakerReview": True,
        "context_hasContextWorkbench": True,
        "context_hasParticipantIdentity": True,
        "api_hasParticipantIdentityBundle": True,
        "api_contextHasParticipantIdentityBundle": True,
        "final_hasFinalPreview": True,
        "final_hasMemoryCandidates": True,
        "final_hasQueuePreviewButton": True,
        "final_urlConversation": "1",
        "final_urlWorkflow": "output",
    }
    if api_speaker_count:
        expected["speakers_hasManualContactInput"] = True
        expected["speakers_hasManualContactButton"] = True
    if api_calendar_attendee_count:
        expected["speakers_hasCalendarEvidence"] = True
    if api_source_profile_count:
        expected["context_hasIdentitySourceProfileChip"] = True
    if api_final_blocked:
        expected["final_hasFinalPreviewBlocked"] = True
        expected["final_hasQueuePreviewButtonDisabled"] = True
    missing = [key for key, expected_value in expected.items() if checks.get(key) != expected_value]
    result = {
        "schema_version": "transcribe-audio.conversation-review-loop-smoke.v2",
        "status": "pass" if not missing else "fail",
        "base_url": base_url,
        "deep_link_url": deep_link_url,
        "session": session,
        "profile": str(profile),
        "viewport": {"width": width, "height": height},
        "selected_document_id": selected_id,
        "identity_summary": {
            "speaker_count": api_speaker_count,
            "candidate_count": len(identity_bundle.get("contact_candidates") or []),
            "calendar_attendee_count": api_calendar_attendee_count,
            "source_profile_count": api_source_profile_count,
            "pending_count": int(identity_review.get("pending_count") or 0),
            "final_blocked_by_identity_or_context": api_final_blocked,
        },
        "checks": checks,
        "expected": expected,
        "missing_checks": missing,
        "screenshot_path": str(screenshot_path),
    }
    report_path = report_dir / f"{stamp}-conversation-review-loop-smoke.json"
    report_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    result["report_path"] = str(report_path)
    if missing:
        raise RuntimeError(f"Conversation review loop smoke failed checks: {', '.join(missing)}")
    return result


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    default_profile = args.report_dir.expanduser() / "profiles" / args.session
    try:
        result = run_smoke(
            base_url=args.base_url,
            session=args.session,
            profile=args.profile or default_profile,
            report_dir=args.report_dir,
            query=args.query,
            kind=args.kind,
            viewport=parse_viewport(args.viewport),
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"{CONVERSATION_REVIEW_LOOP_SMOKE_JSON_STDOUT_PREFIX}{json.dumps(result, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
