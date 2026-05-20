from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import intelligence_config


def test_resolve_task_config_uses_defaults_without_user_file(tmp_path: Path) -> None:
    config = intelligence_config.resolve_task_config(
        intelligence_config.TASK_FIRST_PASS_SUMMARY,
        path=tmp_path / "missing.json",
    )

    assert config.provider == "openai-compatible"
    assert config.model == "gpt-4o-mini"
    assert config.timeout == 120.0
    assert config.source == "defaults"


def test_resolve_task_config_applies_file_env_and_cli_overrides(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "intelligence.config.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": intelligence_config.SCHEMA_VERSION,
                "tasks": {
                    "first_pass_summary": {
                        "provider": "codex-exec",
                        "model": "gpt-file",
                        "timeout": 45,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("TRANSCRIPTS_INTELLIGENCE_FIRST_PASS_SUMMARY_MODEL", "gpt-env")

    config = intelligence_config.resolve_task_config(
        intelligence_config.TASK_FIRST_PASS_SUMMARY,
        path=path,
        overrides={"temperature": 0.2},
    )

    assert config.provider == "codex-exec"
    assert config.model == "gpt-env"
    assert config.timeout == 45
    assert config.temperature == 0.2
    assert config.source == "override"


def test_apply_task_config_updates_arg_namespace(tmp_path: Path) -> None:
    path = tmp_path / "intelligence.config.json"
    path.write_text(
        json.dumps(
            {
                "tasks": {
                    "contextual_reread": {
                        "provider": "codex-exec",
                        "model": "gpt-config",
                        "timeout": 30,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    args = Namespace(
        intelligence_config=path,
        provider=None,
        model=None,
        base_url=None,
        timeout=None,
        temperature=None,
    )

    config = intelligence_config.apply_task_config(args, intelligence_config.TASK_CONTEXTUAL_REREAD)

    assert config.provider == "codex-exec"
    assert args.provider == "codex-exec"
    assert args.model == "gpt-config"
    assert args.timeout == 30
    assert args.intelligence_task == "contextual_reread"


def test_write_sample_config_contains_all_tasks(tmp_path: Path) -> None:
    path = intelligence_config.write_sample_config(tmp_path / "sample.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == intelligence_config.SCHEMA_VERSION
    assert set(payload["tasks"]) == set(intelligence_config.TASK_IDS)
