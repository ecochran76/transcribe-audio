from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import codex_app_server_client


def test_start_model_turn_uses_current_app_server_handshake(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured_path = tmp_path / "messages.jsonl"
    server_script = """
import json
import sys

captured_path = sys.argv[1]
for line in sys.stdin:
    message = json.loads(line)
    with open(captured_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(message) + "\\n")
    method = message.get("method")
    if method == "initialize":
        print(json.dumps({"id": message["id"], "result": {"userAgent": "test"}}), flush=True)
    elif method == "thread/start":
        print(json.dumps({"id": message["id"], "result": {"thread": {"id": "thread-1"}}}), flush=True)
    elif method == "turn/start":
        print(json.dumps({"id": message["id"], "result": {"turn": {"id": "turn-1"}}}), flush=True)
        print(json.dumps({"method": "turn/completed", "params": {"turn": {"id": "turn-1", "status": "completed"}}}), flush=True)
    elif method == "thread/turns/list":
        print(json.dumps({"id": message["id"], "result": {"data": [{"id": "turn-1", "status": "completed", "items": [{"type": "userMessage", "content": [{"type": "text", "text": "prompt"}]}, {"type": "agentMessage", "text": "{\\"ok\\": true}"}]}]}}), flush=True)
    """
    real_popen = subprocess.Popen
    invoked_args: list[list[str]] = []

    def fake_popen(args: list[str], **kwargs: object) -> subprocess.Popen[str]:
        invoked_args.append(args)
        return real_popen(
            [sys.executable, "-u", "-c", server_script, str(captured_path)],
            **kwargs,
        )

    monkeypatch.setattr(codex_app_server_client.subprocess, "Popen", fake_popen)

    result = codex_app_server_client.start_model_turn(
        codex_bin="codex",
        cwd=tmp_path,
        prompt_text="Identify speakers.",
        model="gpt-5.6-sol",
        timeout_seconds=1,
    )

    assert result["thread_id"] == "thread-1"
    assert result["turn_id"] == "turn-1"
    assert invoked_args == [["codex", "app-server", "--listen", "stdio://"]]
    messages = [
        json.loads(line)
        for line in captured_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [message["method"] for message in messages] == [
        "initialize",
        "initialized",
        "thread/start",
        "turn/start",
        "thread/turns/list",
    ]
    assert all("jsonrpc" not in message for message in messages)
    assert "id" not in messages[1]


def test_request_timeout_is_not_blocked_by_silent_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_popen = subprocess.Popen

    def fake_popen(_args: list[str], **kwargs: object) -> subprocess.Popen[str]:
        return real_popen(
            [sys.executable, "-u", "-c", "import time; time.sleep(2)"],
            **kwargs,
        )

    monkeypatch.setattr(codex_app_server_client.subprocess, "Popen", fake_popen)
    client = codex_app_server_client.CodexAppServerClient(
        codex_bin="codex",
        timeout_seconds=0.05,
    )
    started_at = time.monotonic()
    try:
        with pytest.raises(
            codex_app_server_client.CodexAppServerError,
            match="Timed out waiting for initialize response",
        ):
            client.request("initialize", {"clientInfo": {"name": "test", "version": "1"}})
    finally:
        client.close()

    assert time.monotonic() - started_at < 1


def test_inspect_model_turn_resumes_thread_on_new_stdio_connection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured_path = tmp_path / "inspect-messages.jsonl"
    server_script = """
import json
import sys

captured_path = sys.argv[1]
for line in sys.stdin:
    message = json.loads(line)
    with open(captured_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(message) + "\\n")
    if "id" not in message:
        continue
    method = message.get("method")
    if method == "initialize":
        result = {"userAgent": "test"}
    elif method == "thread/resume":
        result = {"thread": {"id": "thread-1"}}
    elif method == "thread/read":
        result = {"thread": {"id": "thread-1"}}
    elif method == "thread/turns/list":
        result = {"data": [{"id": "turn-1", "status": "completed", "items": [{"type": "userMessage", "content": [{"type": "text", "text": "prompt"}]}, {"type": "agentMessage", "text": "{\\"ok\\": true}"}]}]}
    else:
        result = {}
    print(json.dumps({"id": message["id"], "result": result}), flush=True)
"""
    real_popen = subprocess.Popen

    def fake_popen(_args: list[str], **kwargs: object) -> subprocess.Popen[str]:
        return real_popen(
            [sys.executable, "-u", "-c", server_script, str(captured_path)],
            **kwargs,
        )

    monkeypatch.setattr(codex_app_server_client.subprocess, "Popen", fake_popen)

    result = codex_app_server_client.inspect_model_turn(
        codex_bin="codex",
        thread_id="thread-1",
        turn_id="turn-1",
        timeout_seconds=1,
    )

    assert result["completed"] is True
    assert result["output_text"] == '{"ok": true}'
    methods = [
        json.loads(line)["method"]
        for line in captured_path.read_text(encoding="utf-8").splitlines()
    ]
    assert methods[:3] == ["initialize", "initialized", "thread/resume"]
