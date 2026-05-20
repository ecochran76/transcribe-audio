#!/usr/bin/env python3
"""
Minimal JSON-RPC client for the local Codex app-server proxy.
"""
from __future__ import annotations

import json
import subprocess
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
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params or {},
        }
        self.process.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
        self.process.stdin.flush()
        deadline = time.monotonic() + self.timeout_seconds
        while time.monotonic() < deadline:
            line = self.process.stdout.readline()
            if not line:
                break
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
    client = CodexAppServerClient(codex_bin=codex_bin, use_proxy=True, timeout_seconds=timeout_seconds)
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
        if existing_thread_id:
            thread_id = existing_thread_id
            thread_response: dict[str, Any] = {"thread": {"id": thread_id}, "resumed": False}
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
        return {
            "ok": True,
            "thread_id": thread_id,
            "turn_id": turn_id,
            "thread_start_response": thread_response,
            "turn_start_response": turn_response,
            "events": client.events,
        }
    finally:
        client.close()
