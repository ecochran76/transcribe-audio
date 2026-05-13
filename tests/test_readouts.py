from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import summarize_transcript
import contextual_reread
from readout_artifacts import Readout
from watch_transcriptions import (
    JobState,
    ProcessedRecord,
    WatchJob,
    extract_readout_paths,
    load_jobs,
    load_state,
    parse_args,
    save_state,
)


def test_readout_schema_and_markdown() -> None:
    readout = Readout.from_model_payload(
        {
            "title": "Meeting Readout",
            "summary": "A short discussion happened.",
            "participants": [{"name": "Alice", "role": "speaker", "evidence": "calendar"}],
            "topics": ["planning"],
            "action_items": [{"task": "Follow up", "owner": "Alice", "due": "", "status": "open"}],
            "unresolved_questions": ["What is the deadline?"],
            "matter_candidates": [{"label": "Project X", "confidence": 0.7, "evidence": "topic"}],
            "memory_candidates": [{"text": "Alice owns follow-up.", "kind": "task", "evidence": "action item"}],
        },
        source_artifact_path=Path("/tmp/example.transcript.json"),
        provider={"name": "openai-compatible", "model": "test"},
    )

    payload = readout.to_dict()
    markdown = readout.to_markdown()

    assert payload["schema_version"] == 1
    assert payload["summary"] == "A short discussion happened."
    assert payload["topics"] == ["planning"]
    assert "# Meeting Readout" in markdown
    assert "Alice owns follow-up." in markdown


def test_readout_contextualization_renders_supporting_sources() -> None:
    readout = Readout.from_model_payload(
        {
            "title": "Contextual Readout",
            "summary": "Updated summary.",
            "participants": [],
            "topics": [],
            "action_items": [],
            "unresolved_questions": [],
            "matter_candidates": [],
            "memory_candidates": [],
            "risks": [],
            "next_steps": [],
        },
        source_artifact_path=Path("/tmp/example.transcript.json"),
        provider={"name": "codex-exec", "model": "test"},
        contextualization={
            "supporting_context_sources": [
                {"source_type": "odollo_contact", "source_id": "contact-1", "label": "Tempo Chemical"}
            ]
        },
    )

    payload = readout.to_dict()
    markdown = readout.to_markdown()

    assert payload["contextualization"]["supporting_context_sources"][0]["source_type"] == "odollo_contact"
    assert "## Supporting Context Sources" in markdown
    assert "Tempo Chemical" in markdown


def test_extract_readout_paths_from_stdout() -> None:
    stdout = "\n".join(
        [
            "Writing readout JSON...",
            "READOUT_JSON=/tmp/a.readout.json",
            "READOUT_JSON=/tmp/a.readout.json",
            "READOUT_JSON=/tmp/b.readout.json",
        ]
    )

    assert extract_readout_paths(stdout) == ["/tmp/a.readout.json", "/tmp/b.readout.json"]


def test_watcher_readout_config_expands_to_job_args(tmp_path: Path) -> None:
    config_path = tmp_path / "watch.json"
    config_path.write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "name": "downloads",
                        "watch_dir": str(tmp_path),
                        "backend": "faster_whisper",
                        "readout": {
                            "enabled": True,
                            "provider": "openai-compatible",
                            "model": "gpt-test",
                            "base_url": "http://127.0.0.1:9999/v1",
                            "output_dir": str(tmp_path / "readouts"),
                            "timeout": 5,
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    jobs = load_jobs(config_path, parse_args(["--config", str(config_path)]))

    assert jobs[0].readout_enabled is True
    assert jobs[0].readout_args == [
        "--provider",
        "openai-compatible",
        "--model",
        "gpt-test",
        "--base-url",
        "http://127.0.0.1:9999/v1",
        "--output-dir",
        str(tmp_path / "readouts"),
        "--timeout",
        "5",
    ]


def test_watcher_state_preserves_readout_paths(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    job = WatchJob(
        name="downloads",
        watch_dir=tmp_path,
        glob="*.m4a",
        backends=["assembly"],
        recursive=False,
        settle_seconds=120,
        min_age_seconds=20,
        scan_interval=30,
        failure_retry_seconds=900,
        cli_args={"assembly": []},
        notify_on_success=False,
        notify_on_failure=False,
        slack_channel=None,
    )
    readout_path = str(tmp_path / "meeting Transcript.readout.json")
    save_state(
        state_path,
        {
            job.name: JobState(
                processed={
                    str(tmp_path / "meeting.m4a"): ProcessedRecord(
                        status="success",
                        completed_at=1.0,
                        size=123,
                        mtime=2.0,
                        fingerprint="abc",
                        command=["python", "assembly_transcribe.py"],
                        returncode=0,
                        backend="assembly",
                        attempted_backends=["assembly"],
                        artifact_paths=[str(tmp_path / "meeting Transcript.transcript.json")],
                        readout_paths=[readout_path],
                    )
                },
                candidates={},
            )
        },
    )

    loaded = load_state(state_path, [job])
    record = next(iter(loaded[job.name].processed.values()))
    assert record.readout_paths == [readout_path]


def test_openai_compatible_readout_call(monkeypatch) -> None:
    captured = {}

    class FakeResponse:
        status_code = 200

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "title": "Readout",
                                    "summary": "Summary text.",
                                    "participants": [],
                                    "topics": ["topic"],
                                    "action_items": [],
                                    "unresolved_questions": [],
                                    "matter_candidates": [],
                                    "memory_candidates": [],
                                    "risks": [],
                                    "next_steps": [],
                                }
                            )
                        }
                    }
                ]
            }

    class FakeSession:
        def __init__(self):
            self.headers = {}

        def post(self, endpoint, *, json, timeout):
            captured["endpoint"] = endpoint
            captured["payload"] = json
            captured["timeout"] = timeout
            return FakeResponse()

    monkeypatch.setattr(summarize_transcript.requests, "Session", FakeSession)

    result = summarize_transcript.call_openai_compatible(
        {"transcript_text": "Speaker: hello"},
        api_key="test-key",
        base_url="http://127.0.0.1:9999/v1",
        model="gpt-test",
        timeout=3,
        temperature=0,
    )

    assert result["summary"] == "Summary text."
    assert captured["endpoint"] == "http://127.0.0.1:9999/v1/chat/completions"
    assert captured["payload"]["response_format"] == {"type": "json_object"}


def test_openai_compatible_readout_request_error_is_clean(monkeypatch) -> None:
    class FakeSession:
        def __init__(self):
            self.headers = {}

        def post(self, endpoint, *, json, timeout):
            raise summarize_transcript.requests.ReadTimeout("slow provider")

    monkeypatch.setattr(summarize_transcript.requests, "Session", FakeSession)

    try:
        summarize_transcript.call_openai_compatible(
            {"transcript_text": "Speaker: hello"},
            api_key="test-key",
            base_url="http://127.0.0.1:9999/v1",
            model="gpt-test",
            timeout=3,
            temperature=0,
        )
    except summarize_transcript.TranscriptionError as exc:
        assert "OpenAI-compatible readout request failed" in str(exc)
    else:
        raise AssertionError("Expected TranscriptionError")


def test_openai_compatible_readout_extracts_wrapped_json(monkeypatch) -> None:
    class FakeResponse:
        status_code = 200

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                "Here is the readout:\n\n"
                                "```json\n"
                                '{"title":"Wrapped","summary":"Wrapped summary.","participants":[],"topics":[],'
                                '"action_items":[],"unresolved_questions":[],"matter_candidates":[],'
                                '"memory_candidates":[],"risks":[],"next_steps":[]}'
                                "\n```"
                            )
                        }
                    }
                ]
            }

    class FakeSession:
        def __init__(self):
            self.headers = {}

        def post(self, endpoint, *, json, timeout):
            return FakeResponse()

    monkeypatch.setattr(summarize_transcript.requests, "Session", FakeSession)

    result = summarize_transcript.call_openai_compatible(
        {"transcript_text": "Speaker: hello"},
        api_key="test-key",
        base_url="http://127.0.0.1:9999/v1",
        model="gpt-test",
        timeout=3,
        temperature=0,
    )

    assert result["title"] == "Wrapped"
    assert result["summary"] == "Wrapped summary."


def test_openai_compatible_readout_rejects_empty_shape(monkeypatch) -> None:
    class FakeResponse:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": '{"instructions":"echoed input"}'}}]}

    class FakeSession:
        def __init__(self):
            self.headers = {}

        def post(self, endpoint, *, json, timeout):
            return FakeResponse()

    monkeypatch.setattr(summarize_transcript.requests, "Session", FakeSession)

    try:
        summarize_transcript.call_openai_compatible(
            {"transcript_text": "Speaker: hello"},
            api_key="test-key",
            base_url="http://127.0.0.1:9999/v1",
            model="gpt-test",
            timeout=3,
            temperature=0,
        )
    except summarize_transcript.TranscriptionError as exc:
        assert "without readout content" in str(exc)
    else:
        raise AssertionError("Expected TranscriptionError")


def test_codex_exec_readout_call(monkeypatch) -> None:
    captured = {}

    def fake_run(command, *, input, text, capture_output, timeout, check):
        captured["command"] = command
        captured["input"] = input
        captured["timeout"] = timeout
        output_path = Path(command[command.index("--output-last-message") + 1])
        output_path.write_text(
            json.dumps(
                {
                    "title": "Codex Readout",
                    "summary": "Codex summary.",
                    "participants": [],
                    "topics": ["topic"],
                    "action_items": [],
                    "unresolved_questions": [],
                    "matter_candidates": [],
                    "memory_candidates": [],
                    "risks": [],
                    "next_steps": [],
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(summarize_transcript.subprocess, "run", fake_run)

    result = summarize_transcript.call_codex_exec(
        {"transcript_text": "Speaker: hello"},
        model="gpt-test",
        timeout=3,
    )

    assert result["summary"] == "Codex summary."
    assert captured["command"][:4] == ["codex", "--ask-for-approval", "never", "exec"]
    assert "--sandbox" in captured["command"]
    assert "read-only" in captured["command"]
    assert "--model" in captured["command"]
    assert "Return ONLY one valid JSON object" in captured["input"]


def test_readout_prompt_includes_calendar_context() -> None:
    prompt = summarize_transcript.build_readout_prompt(
        {
            "transcript_title": "Transcript",
            "event": {
                "summary": "SIP WMA Hamburg",
                "participants": ["eric@example.com"],
                "matching_calendars": [
                    {
                        "calendar_id": "primary",
                        "calendar_summary": "Eric - SoyLei",
                        "event_summary": "SIP WMA Hamburg",
                        "coverage": 0.74,
                    }
                ],
            },
            "transcript_text": "Speaker: hello",
        }
    )

    payload = json.loads(prompt)

    assert "Return ONLY one valid JSON object" in payload["instructions"]
    assert payload["calendar_context"]["primary_event_summary"] == "SIP WMA Hamburg"
    assert payload["calendar_context"]["primary_event_participants"] == ["eric@example.com"]
    assert payload["calendar_context"]["matching_calendars"][0]["calendar_summary"] == "Eric - SoyLei"
    assert "context evidence" in payload["calendar_context"]["calendar_context_guidance"]


def test_readout_prompt_includes_contextual_reread_context() -> None:
    prompt = summarize_transcript.build_readout_prompt(
        {
            "transcript_text": "Speaker: hello",
            "prior_readout": {"summary": "Initial summary."},
            "route_decision": {"selected_candidate": {"label": "Tempo matter"}},
            "supporting_context": {
                "sources": [
                    {
                        "source_type": "odollo_contact",
                        "source_id": "contact-1",
                        "label": "Tempo Chemical",
                    }
                ]
            },
        }
    )

    payload = json.loads(prompt)

    assert payload["prior_readout"]["summary"] == "Initial summary."
    assert payload["route_decision"]["selected_candidate"]["label"] == "Tempo matter"
    assert payload["supporting_context"]["sources"][0]["source_type"] == "odollo_contact"


def test_system_prompt_mentions_matching_calendars() -> None:
    system_prompt = summarize_transcript.readout_system_prompt()

    assert "calendar_context.matching_calendars" in system_prompt
    assert "supporting_context" in system_prompt
    assert "matter_candidates" in system_prompt


def test_contextual_reread_supporting_context_prefers_selected_and_calendar_sources() -> None:
    route = {
        "status": "selected",
        "review_required": False,
        "selected_candidate": {
            "label": "Tempo matter",
            "provenance_source_ids": ["odollo-contact-1"],
        },
        "provenance_pack": {
            "excluded_sources": [{"source_type": "odollo_contact", "source_id": "bad-1", "label": "Unrelated"}],
            "sources": [
                {"source_type": "graphiti_node", "source_id": "node-1", "label": "Graphiti matter"},
                {"source_type": "odollo_contact", "source_id": "odollo-contact-1", "label": "Tempo Chemical"},
                {"source_type": "gws_calendar_overlap", "source_id": "cal-1", "label": "SoyLei calendar"},
            ]
        },
        "warnings": ["Excluded 1 provenance source(s) below quality threshold 2."],
    }

    context = contextual_reread.build_supporting_context(route, max_sources=2, snippet_chars=80)

    assert [source["source_id"] for source in context["sources"]] == ["odollo-contact-1", "cal-1"]
    assert context["selected_candidate"]["label"] == "Tempo matter"
    assert context["excluded_source_count"] == 1
    assert context["warnings"] == ["Excluded 1 provenance source(s) below quality threshold 2."]


def test_contextual_reread_generate_uses_provider_and_writes_outputs(monkeypatch, tmp_path: Path) -> None:
    transcript_path = tmp_path / "meeting.transcript.json"
    readout_path = tmp_path / "meeting.readout.json"
    route_path = tmp_path / "meeting.route.json"
    output_dir = tmp_path / "out"
    transcript_path.write_text(json.dumps({"transcript_text": "Speaker: hello"}), encoding="utf-8")
    readout_path.write_text(json.dumps({"summary": "Initial summary.", "matter_candidates": []}), encoding="utf-8")
    route_path.write_text(
        json.dumps(
            {
                "status": "selected",
                "review_required": False,
                "selected_candidate": {"label": "Tempo matter", "provenance_source_ids": ["contact-1"]},
                "provenance_pack": {
                    "sources": [
                        {"source_type": "odollo_contact", "source_id": "contact-1", "label": "Tempo Chemical"}
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    def fake_provider(args, artifact, provider_info):
        assert artifact["prior_readout"]["summary"] == "Initial summary."
        assert artifact["supporting_context"]["sources"][0]["label"] == "Tempo Chemical"
        return {
            "title": "Contextual Readout",
            "summary": "Updated summary.",
            "participants": [],
            "topics": [],
            "action_items": [],
            "unresolved_questions": [],
            "matter_candidates": [],
            "memory_candidates": [],
            "risks": [],
            "next_steps": [],
        }

    monkeypatch.setattr(contextual_reread, "call_provider", fake_provider)

    json_path, markdown_path = contextual_reread.generate_contextual_readout(
        contextual_reread.parse_args(
            [
                str(transcript_path),
                str(readout_path),
                str(route_path),
                "--output-dir",
                str(output_dir),
                "--provider",
                "codex-exec",
            ]
        )
    )

    payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert json_path.name == "meeting.contextual.readout.json"
    assert markdown_path.name == "meeting.contextual.readout.md"
    assert payload["summary"] == "Updated summary."
    assert payload["contextualization"]["supporting_context_sources"][0]["source_id"] == "contact-1"
