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


def test_codex_app_server_defaults_to_current_workstation_model(tmp_path: Path) -> None:
    speaker_config = intelligence_config.resolve_task_config(
        intelligence_config.TASK_SPEAKER_DISAMBIGUATION,
        path=tmp_path / "missing.json",
    )
    supervisor_config = intelligence_config.resolve_task_config(
        intelligence_config.TASK_APP_SUPERVISOR,
        path=tmp_path / "missing.json",
    )

    assert speaker_config.provider == "codex-app-server"
    assert speaker_config.model == "gpt-5.6-sol"
    assert supervisor_config.provider == "codex-app-server"
    assert supervisor_config.model == "gpt-5.6-sol"


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
    assert set(payload["profiles"]) == set(intelligence_config.DEFAULT_PROFILES)
    assert set(payload["task_profiles"]) == set(intelligence_config.DEFAULT_TASK_PROFILES)
    assert set(payload["tasks"]) == set(intelligence_config.TASK_IDS)
    assert "provider" not in payload["tasks"]["first_pass_summary"]
    assert "model" not in payload["tasks"]["first_pass_summary"]
    assert payload["tasks"]["first_pass_summary"]["human_review"] == "on_warning"


def test_preview_config_update_does_not_write(tmp_path: Path) -> None:
    path = tmp_path / "intelligence.config.json"

    preview = intelligence_config.preview_config_update(
        task="first_pass_summary",
        update={"provider": "codex-exec", "model": "gpt-test"},
        path=path,
    )

    assert preview["will_write"] is False
    assert preview["rollback"]["delete_task"] is True
    assert preview["rollback"]["previous_task_config"] == {}
    assert preview["after"]["tasks"]["first_pass_summary"]["provider"] == "codex-exec"
    assert preview["resolved_after"]["model"] == "gpt-test"
    assert not path.exists()


def test_profile_assignment_resolves_task_through_profile(tmp_path: Path) -> None:
    path = tmp_path / "intelligence.config.json"
    path.write_text(
        json.dumps(
            {
                "profiles": {
                    "extended_readout": {
                        "label": "Extended readout",
                        "provider": "openai-compatible",
                        "model": "gpt-extended",
                        "timeout": 300,
                        "temperature": 0.2,
                    }
                },
                "task_profiles": {"first_pass_summary": "extended_readout"},
                "tasks": {
                    "first_pass_summary": {
                        "fallbacks": ["codex-exec"],
                        "human_review": "on_warning",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    resolved = intelligence_config.resolve_task_config("first_pass_summary", path=path)

    assert resolved.profile == "extended_readout"
    assert resolved.provider == "openai-compatible"
    assert resolved.model == "gpt-extended"
    assert resolved.timeout == 300
    assert resolved.temperature == 0.2
    assert resolved.human_review == "on_warning"


def test_preview_profile_and_task_profile_update(tmp_path: Path) -> None:
    path = tmp_path / "intelligence.config.json"

    preview = intelligence_config.preview_config_update(
        task="first_pass_summary",
        update={"profile": "openai_readout", "human_review": "required"},
        profile_id="openai_readout",
        profile_update={"model": "gpt-profiled", "timeout": 240},
        path=path,
    )

    assert preview["will_write"] is False
    assert preview["after"]["profiles"]["openai_readout"]["model"] == "gpt-profiled"
    assert preview["after"]["task_profiles"]["first_pass_summary"] == "openai_readout"
    assert preview["after"]["tasks"]["first_pass_summary"]["human_review"] == "required"
    assert "provider" not in preview["after"]["tasks"]["first_pass_summary"]
    assert preview["resolved_after"]["model"] == "gpt-profiled"
    assert not path.exists()


def test_profile_only_preview_and_apply_do_not_require_task(tmp_path: Path) -> None:
    path = tmp_path / "intelligence.config.json"

    preview = intelligence_config.preview_config_update(
        profile_id="openai_readout",
        profile_update={"label": "Extended Pro readout", "model": "gpt-extended-pro"},
        path=path,
    )

    assert preview["task"] == ""
    assert preview["resolved_after"] is None
    assert preview["after"]["profiles"]["openai_readout"]["label"] == "Extended Pro readout"
    assert preview["after"]["profiles"]["openai_readout"]["model"] == "gpt-extended-pro"
    assert "task" not in preview["rollback"]
    assert not path.exists()

    applied = intelligence_config.apply_config_update(
        profile_id="openai_readout",
        profile_update={"label": "Extended Pro readout", "model": "gpt-extended-pro"},
        approval_token=intelligence_config.APPLY_APPROVAL_TOKEN,
        path=path,
    )

    assert applied["will_write"] is True
    assert applied["resolved_after"] is None
    assert path.exists()
    resolved = intelligence_config.resolve_task_config("first_pass_summary", path=path)
    assert resolved.model == "gpt-extended-pro"


def test_delete_custom_profile_requires_no_task_reference(tmp_path: Path) -> None:
    path = tmp_path / "intelligence.config.json"
    path.write_text(
        json.dumps(
            {
                "profiles": {
                    "custom_readout": {
                        "label": "Custom readout",
                        "provider": "openai-compatible",
                        "model": "gpt-custom",
                    }
                },
                "task_profiles": {"first_pass_summary": "custom_readout"},
            }
        ),
        encoding="utf-8",
    )

    try:
        intelligence_config.preview_config_update(
            profile_id="custom_readout",
            delete_profile=True,
            path=path,
        )
    except ValueError as exc:
        assert "assigned to" in str(exc)
    else:
        raise AssertionError("profile deletion must fail while tasks reference it")

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["task_profiles"] = {}
    path.write_text(json.dumps(payload), encoding="utf-8")

    preview = intelligence_config.preview_config_update(
        profile_id="custom_readout",
        delete_profile=True,
        path=path,
    )
    assert preview["delete_profile"] is True
    assert "custom_readout" not in preview["after"]["profiles"]
    assert path.exists()

    applied = intelligence_config.apply_config_update(
        profile_id="custom_readout",
        delete_profile=True,
        approval_token=intelligence_config.APPLY_APPROVAL_TOKEN,
        path=path,
    )
    assert applied["applied"] is True
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert "custom_readout" not in stored["profiles"]


def test_delete_default_profile_is_rejected(tmp_path: Path) -> None:
    try:
        intelligence_config.preview_config_update(
            profile_id="openai_readout",
            delete_profile=True,
            path=tmp_path / "intelligence.config.json",
        )
    except ValueError as exc:
        assert "Default intelligence profiles cannot be deleted" in str(exc)
    else:
        raise AssertionError("default profile deletion must fail")


def test_apply_config_update_requires_token_and_writes(tmp_path: Path) -> None:
    path = tmp_path / "intelligence.config.json"

    try:
        intelligence_config.apply_config_update(
            task="first_pass_summary",
            update={"provider": "codex-exec"},
            approval_token="wrong",
            path=path,
        )
    except ValueError as exc:
        assert "approval_token" in str(exc)
    else:
        raise AssertionError("apply without approval token must fail")

    applied = intelligence_config.apply_config_update(
        task="first_pass_summary",
        update={"provider": "codex-exec", "fallbacks": ["openai-compatible"]},
        approval_token=intelligence_config.APPLY_APPROVAL_TOKEN,
        path=path,
    )

    stored = json.loads(path.read_text(encoding="utf-8"))
    assert applied["applied"] is True
    assert stored["tasks"]["first_pass_summary"]["provider"] == "codex-exec"
    assert intelligence_config.resolve_task_config("first_pass_summary", path=path).provider == "codex-exec"


def test_preview_config_update_rejects_unknown_fields(tmp_path: Path) -> None:
    try:
        intelligence_config.preview_config_update(
            task="first_pass_summary",
            update={"provider": "codex-exec", "secret": "nope"},
            path=tmp_path / "config.json",
        )
    except ValueError as exc:
        assert "Unknown task config field" in str(exc)
    else:
        raise AssertionError("unknown fields must fail validation")
