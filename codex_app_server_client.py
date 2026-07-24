#!/usr/bin/env python3
"""
Minimal JSON-RPC client for the local Codex app-server proxy.
"""
from __future__ import annotations

import json
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Optional


class CodexAppServerError(RuntimeError):
    pass


class CodexAppServerClient:
    def __init__(
        self,
        *,
        codex_bin: str = "codex",
        use_proxy: bool = True,
        timeout_seconds: float = 30,
    ) -> None:
        self.codex_bin = codex_bin
        self.use_proxy = use_proxy
        self.timeout_seconds = timeout_seconds
        args = [codex_bin, "app-server", "proxy"] if use_proxy else [codex_bin, "app-server", "--listen", "stdio://"]
        self.process = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self.next_id = 1
        self.events: list[dict[str, Any]] = []
        self.stderr = ""
        self._stdout_eof = object()
        self._stdout_lines: queue.Queue[object] = queue.Queue()
        self._stdout_thread = threading.Thread(
            target=self._read_stdout,
            name="codex-app-server-stdout",
            daemon=True,
        )
        self._stdout_thread.start()

    def _read_stdout(self) -> None:
        if self.process.stdout is None:
            self._stdout_lines.put(self._stdout_eof)
            return
        try:
            for line in self.process.stdout:
                self._stdout_lines.put(line)
        finally:
            self._stdout_lines.put(self._stdout_eof)

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()

    def request(self, method: str, params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        if self.process.stdin is None or self.process.stdout is None:
            raise CodexAppServerError("Codex app-server process pipes are unavailable.")
        request_id = self.next_id
        self.next_id += 1
        message = {
            "id": request_id,
            "method": method,
            "params": params or {},
        }
        self._write_message(message)
        deadline = time.monotonic() + self.timeout_seconds
        while time.monotonic() < deadline:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                item = self._stdout_lines.get(timeout=remaining)
            except queue.Empty:
                break
            if item is self._stdout_eof:
                break
            line = str(item)
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and payload.get("id") == request_id:
                if payload.get("error"):
                    raise CodexAppServerError(str(payload["error"]))
                result = payload.get("result")
                return result if isinstance(result, dict) else {"result": result}
            if isinstance(payload, dict) and payload.get("method"):
                self.events.append(payload)
        self._capture_stderr()
        raise CodexAppServerError(f"Timed out waiting for {method} response: {self.stderr[:500]}")

    def notify(self, method: str, params: Optional[dict[str, Any]] = None) -> None:
        self._write_message({"method": method, "params": params or {}})

    def wait_for_turn_completion(self, turn_id: str) -> dict[str, Any]:
        deadline = time.monotonic() + self.timeout_seconds
        while time.monotonic() < deadline:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                item = self._stdout_lines.get(timeout=remaining)
            except queue.Empty:
                break
            if item is self._stdout_eof:
                break
            try:
                payload = json.loads(str(item))
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict) or not payload.get("method"):
                continue
            self.events.append(payload)
            params = payload.get("params") if isinstance(payload.get("params"), dict) else {}
            turn = params.get("turn") if isinstance(params.get("turn"), dict) else {}
            event_turn_id = turn.get("id") or turn.get("turnId")
            if payload.get("method") == "turn/completed" and event_turn_id == turn_id:
                return payload
        self._capture_stderr()
        raise CodexAppServerError(
            f"Timed out waiting for turn/completed notification: {self.stderr[:500]}"
        )

    def _write_message(self, message: dict[str, Any]) -> None:
        if self.process.stdin is None:
            raise CodexAppServerError("Codex app-server process stdin is unavailable.")
        self.process.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
        self.process.stdin.flush()

    def _capture_stderr(self) -> None:
        if self.process.stderr is None:
            return
        try:
            # Avoid blocking on healthy long-lived proxy processes.
            if self.process.poll() is not None:
                self.stderr += self.process.stderr.read()
        except OSError:
            return


def text_input(text: str) -> list[dict[str, str]]:
    return [{"type": "text", "text": text}]


def extract_thread_id(response: dict[str, Any]) -> str:
    thread = response.get("thread") if isinstance(response.get("thread"), dict) else {}
    thread_id = thread.get("id") or thread.get("threadId")
    if not isinstance(thread_id, str) or not thread_id:
        raise CodexAppServerError("app-server response did not include a thread id.")
    return thread_id


def extract_turn_id(response: dict[str, Any]) -> str:
    turn = response.get("turn") if isinstance(response.get("turn"), dict) else {}
    turn_id = turn.get("id") or turn.get("turnId")
    if not isinstance(turn_id, str) or not turn_id:
        raise CodexAppServerError("app-server response did not include a turn id.")
    return turn_id


def start_model_turn(
    *,
    codex_bin: str,
    cwd: Path,
    prompt_text: str,
    model: str = "",
    existing_thread_id: str = "",
    timeout_seconds: float = 30,
) -> dict[str, Any]:
    client = CodexAppServerClient(codex_bin=codex_bin, use_proxy=False, timeout_seconds=timeout_seconds)
    try:
        client.request(
            "initialize",
            {
                "clientInfo": {
                    "name": "transcribe-audio",
                    "version": "0.1.0",
                    "title": "Transcript App Intelligence",
                },
                "capabilities": {"experimentalApi": True},
            },
        )
        client.notify("initialized")
        if existing_thread_id:
            thread_id = existing_thread_id
            thread_response = client.request("thread/resume", {"threadId": thread_id})
        else:
            thread_params: dict[str, Any] = {
                "cwd": str(cwd),
                "approvalPolicy": "on-request",
                "ephemeral": False,
                "developerInstructions": (
                    "You are a bounded worker inside the transcribe-audio App Intelligence supervisor. "
                    "Return analysis only. Do not execute external writes, memory writes, repository writes, "
                    "routing applies, or deposition actions."
                ),
            }
            if model:
                thread_params["model"] = model
            thread_response = client.request("thread/start", thread_params)
            thread_id = extract_thread_id(thread_response)

        turn_params: dict[str, Any] = {
            "threadId": thread_id,
            "cwd": str(cwd),
            "approvalPolicy": "on-request",
            "input": text_input(prompt_text),
        }
        if model:
            turn_params["model"] = model
        turn_response = client.request("turn/start", turn_params)
        turn_id = extract_turn_id(turn_response)
        completion_event = client.wait_for_turn_completion(turn_id)
        turns = client.request(
            "thread/turns/list",
            {
                "threadId": thread_id,
                "itemsView": "full",
                "limit": 25,
                "sortDirection": "desc",
            },
        )
        completed_turn = find_turn(turns, turn_id)
        items = (
            completed_turn.get("items")
            if isinstance(completed_turn.get("items"), list)
            else []
        )
        return {
            "ok": True,
            "thread_id": thread_id,
            "turn_id": turn_id,
            "completed": True,
            "completion_event": completion_event,
            "output_text": extract_output_text(items),
            "thread_start_response": thread_response,
            "turn_start_response": turn_response,
            "turns_list_response": turns,
            "items_list_response": {"data": items},
            "events": client.events,
        }
    finally:
        client.close()


def inspect_model_turn(
    *,
    codex_bin: str,
    thread_id: str,
    turn_id: str,
    timeout_seconds: float = 30,
) -> dict[str, Any]:
    client = CodexAppServerClient(codex_bin=codex_bin, use_proxy=False, timeout_seconds=timeout_seconds)
    try:
        client.request(
            "initialize",
            {
                "clientInfo": {
                    "name": "transcribe-audio",
                    "version": "0.1.0",
                    "title": "Transcript App Intelligence",
                },
                "capabilities": {"experimentalApi": True},
            },
        )
        client.notify("initialized")
        client.request("thread/resume", {"threadId": thread_id})
        thread_read = client.request("thread/read", {"threadId": thread_id, "includeTurns": False})
        turns = client.request(
            "thread/turns/list",
            {
                "threadId": thread_id,
                "itemsView": "full",
                "limit": 25,
                "sortDirection": "desc",
            },
        )
        turn = find_turn(turns, turn_id)
        items = turn.get("items") if isinstance(turn.get("items"), list) else []
        output_text = extract_output_text(items)
        status = str(turn.get("status") or "") if isinstance(turn, dict) else ""
        return {
            "ok": True,
            "thread_id": thread_id,
            "turn_id": turn_id,
            "status": status,
            "completed": status == "completed",
            "output_text": output_text,
            "thread_read_response": thread_read,
            "turns_list_response": turns,
            "items_list_response": {"data": items},
            "events": client.events,
        }
    finally:
        client.close()


def find_turn(turns_response: dict[str, Any], turn_id: str) -> dict[str, Any]:
    turns = turns_response.get("data") if isinstance(turns_response.get("data"), list) else []
    for turn in turns:
        if isinstance(turn, dict) and (turn.get("id") == turn_id or turn.get("turnId") == turn_id):
            return turn
    return {}


def extract_output_text(items: list[Any]) -> str:
    chunks: list[str] = []
    agent_items = [
        item
        for item in items
        if isinstance(item, dict)
        and str(item.get("type") or "") in {"agentMessage", "assistantMessage"}
    ]
    for item in agent_items or items:
        if isinstance(item, dict):
            collect_text_fields(item, chunks)
    return "\n".join(chunk for chunk in chunks if chunk.strip()).strip()


def collect_text_fields(value: Any, chunks: list[str]) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if key in {"text", "content", "message", "finalMessage", "final_message"} and isinstance(nested, str):
                chunks.append(nested)
            else:
                collect_text_fields(nested, chunks)
    elif isinstance(value, list):
        for nested in value:
            collect_text_fields(nested, chunks)
