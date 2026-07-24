from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

import provenance_config
from transcribe_common import CalendarProviderConfig


def write_config(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": provenance_config.SCHEMA_VERSION,
                "active_profile": "default",
                "profiles": {
                    "default": {
                        "source_ids": ["gog-main", "gws-work", "odollo-soylei", "ical-saber"],
                        "workflows": {
                            "calendar_metadata": {
                                "enabled": True,
                                "primary": {"source_id": "gog-main", "calendar_id": "primary"},
                                "provenance_sources": [
                                    {"source_id": "gog-main", "calendar_ids": ["shared@example.com"]},
                                    {"source_id": "gws-work", "calendar_ids": ["team@example.com"]},
                                    {"source_id": "ical-saber"},
                                ],
                            },
                            "participant_identity": {
                                "enabled": True,
                                "source_ids": ["gws-work", "odollo-soylei"],
                            },
                            "context_workbench": {
                                "enabled": True,
                                "source_ids": ["gws-work", "odollo-soylei"],
                            },
                        },
                    }
                },
                "sources": {
                    "gog-main": {
                        "kind": "gog",
                        "enabled": True,
                        "label": "Primary gog",
                        "account": "me@example.com",
                        "client": "work",
                        "capabilities": ["calendar"],
                        "read_only": True,
                    },
                    "gws-work": {
                        "kind": "gws",
                        "enabled": True,
                        "label": "Work gws",
                        "config_dir": "~/.config/gws-work",
                        "capabilities": ["calendar", "people", "gmail"],
                        "gmail": {"page_size": 2},
                        "people": {"surfaces": ["contacts", "other_contacts"], "limit": 4, "query_limit": 6},
                        "read_only": True,
                    },
                    "odollo-soylei": {
                        "kind": "odollo",
                        "enabled": True,
                        "label": "SoyLei Odoo",
                        "tenant_profile": "soylei-prod",
                        "repo_root": "~/workspace.local/odollo",
                        "config_path": "~/.odollo/odollo.yml",
                        "command": ["odollo"],
                        "models": ["res.partner", "crm.lead", "mail.message"],
                        "limits": {"contacts": 3},
                        "read_only": True,
                    },
                    "ical-saber": {
                        "kind": "ical_calendar",
                        "enabled": True,
                        "label": "SABER Zoho",
                        "url_ref": "env:SABER_ICAL_URL",
                        "capabilities": ["calendar"],
                        "read_only": True,
                        "sensitive_fields": ["url_ref"],
                    },
                },
                "mutation_policy": {"audit_dir": str(path.parent / "audit")},
            }
        ),
        encoding="utf-8",
    )


def test_calendar_settings_resolve_sources_and_redact_ical(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "provenance.config.json"
    write_config(path)
    monkeypatch.setenv("SABER_ICAL_URL", "https://calendar.example.invalid/private-feed")

    settings = provenance_config.resolve_calendar_metadata_settings(path=path)

    assert [config.name for config in settings.provider_configs] == ["gog", "gws"]
    assert settings.provider_configs[0].account == "me@example.com"
    assert settings.provenance_calendar_ids == ["shared@example.com", "team@example.com"]
    assert settings.provenance_ical_urls == ["SABER Zoho=https://calendar.example.invalid/private-feed"]
    assert settings.to_dict()["provenance_ical_urls"] == ["SABER Zoho=[redacted]"]
    assert "private-feed" not in json.dumps(provenance_config.redacted_config(provenance_config.read_config(path)))


def test_calendar_settings_apply_to_cli_args_and_preserve_overrides(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "provenance.config.json"
    write_config(path)
    monkeypatch.setenv("SABER_ICAL_URL", "https://calendar.example.invalid/private-feed")
    args = Namespace(
        provenance_config=path,
        provenance_profile=None,
        calendar_id="primary",
        calendar_providers=None,
        calendar_provenance_calendar_ids=["manual@example.com"],
        calendar_provenance_ical_urls=["Manual=https://calendar.example.invalid/manual"],
    )

    provider_configs = provenance_config.configured_provider_configs_or_fallback(
        args,
        [CalendarProviderConfig(name="google-api")],
    )

    assert [config.name for config in provider_configs] == ["gog", "gws"]
    assert args.calendar_provenance_calendar_ids == ["shared@example.com", "team@example.com", "manual@example.com"]
    assert args.calendar_provenance_ical_urls == [
        "SABER Zoho=https://calendar.example.invalid/private-feed",
        "Manual=https://calendar.example.invalid/manual",
    ]


def test_contact_source_config_resolves_gws_and_odollo_profiles(tmp_path: Path) -> None:
    path = tmp_path / "provenance.config.json"
    write_config(path)

    config = provenance_config.contact_source_config_from_provenance(path=path)

    assert config["gws"]["profiles"][0]["label"] == "Work gws"
    assert config["gws"]["profiles"][0]["surfaces"] == ["contacts", "other_contacts"]
    assert config["odollo"]["profiles"][0]["label"] == "soylei-prod"
    assert config["odollo"]["profiles"][0]["command"] == ["odollo"]
    assert config["odollo"]["profiles"][0]["limit"] == 3


def test_all_config_includes_resolved_contact_source_config(tmp_path: Path) -> None:
    path = tmp_path / "provenance.config.json"
    write_config(path)

    payload = provenance_config.all_config(path=path)

    assert payload["contact_source_config"]["gws"]["profiles"][0]["label"] == "Work gws"
    assert payload["contact_source_config"]["odollo"]["profiles"][0]["label"] == "soylei-prod"


def test_context_source_configs_resolve_adapter_configs(tmp_path: Path) -> None:
    path = tmp_path / "provenance.config.json"
    write_config(path)

    config = provenance_config.context_source_configs_from_provenance(path=path)

    assert config["gws"][0].profile_label == "Work gws"
    assert config["gws"][0].include_drive_search is False
    assert config["gws"][0].include_gmail_search is True
    assert config["gws"][0].gmail_page_size == 2
    assert config["gws"][0].include_people_contacts is True
    assert config["odollo"][0].profiles == ("soylei-prod",)
    assert config["odollo"][0].include_contacts is True
    assert config["odollo"][0].include_leads is True
    assert config["odollo"][0].include_log_notes is True


def test_speaker_preprocessing_excludes_sources_without_source_context(tmp_path: Path) -> None:
    path = tmp_path / "provenance.config.json"
    write_config(path)

    config = provenance_config.speaker_preprocessing_source_configs_from_provenance(path=path)

    assert config["gws"] == []
    assert config["odollo"] == []
    assert config["source_contexts"] == []
    assert config["warnings"] == [
        "Speaker preprocessing excluded source gws-work: missing Source Context.",
        "Speaker preprocessing excluded source odollo-soylei: missing Source Context.",
    ]
    assert provenance_config.validate_config(provenance_config.read_config(path))["valid"] is True


def test_speaker_preprocessing_returns_bounded_semantic_source_context(tmp_path: Path) -> None:
    path = tmp_path / "provenance.config.json"
    write_config(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["sources"]["gws-work"]["source_context"] = {
        "owner": {"type": "person", "id": "operator", "label": "Local operator"},
        "relationship_scope": "personal",
        "account_label": "Personal workspace",
        "evidence_capabilities": ["gmail", "people"],
        "authoritative_identifiers": ["email"],
    }
    raw["sources"]["odollo-soylei"]["source_context"] = {
        "owner": {"type": "organization", "id": "org-soylei", "label": "SoyLei"},
        "relationship_scope": "company",
        "account_label": "SoyLei production",
        "evidence_capabilities": ["contacts", "leads", "log_notes"],
        "authoritative_identifiers": [],
    }
    path.write_text(json.dumps(raw), encoding="utf-8")

    config = provenance_config.speaker_preprocessing_source_configs_from_provenance(path=path)

    assert len(config["gws"]) == 1
    assert len(config["odollo"]) == 1
    assert config["warnings"] == []
    assert config["source_contexts"] == [
        {"source_id": "gws-work", **raw["sources"]["gws-work"]["source_context"]},
        {"source_id": "odollo-soylei", **raw["sources"]["odollo-soylei"]["source_context"]},
    ]
    assert "config_dir" not in json.dumps(config["source_contexts"])
    assert "command" not in json.dumps(config["source_contexts"])


def test_speaker_preprocessing_excludes_incomplete_source_context(tmp_path: Path) -> None:
    path = tmp_path / "provenance.config.json"
    write_config(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["sources"]["gws-work"]["source_context"] = {
        "owner": {"type": "person", "id": "operator", "label": "Local operator"},
        "relationship_scope": "",
        "account_label": "Personal workspace",
        "evidence_capabilities": ["gmail"],
        "authoritative_identifiers": ["email"],
    }
    path.write_text(json.dumps(raw), encoding="utf-8")

    config = provenance_config.speaker_preprocessing_source_configs_from_provenance(path=path)

    assert config["gws"] == []
    assert config["source_contexts"] == []
    assert config["warnings"][0] == (
        "Speaker preprocessing excluded source gws-work: "
        "invalid Source Context (relationship_scope is required)."
    )


def test_preview_and_apply_update_redact_sensitive_values(tmp_path: Path) -> None:
    path = tmp_path / "provenance.config.json"
    update = {
        "active_profile": "default",
        "profiles": {"default": {"source_ids": ["ical-private"]}},
        "sources": {
            "ical-private": {
                "kind": "ical_calendar",
                "enabled": True,
                "label": "Private feed",
                "url": "https://calendar.example.invalid/private-secret",
                "read_only": True,
            }
        },
    }

    preview = provenance_config.preview_config_update(update=update, path=path)

    assert preview["will_write"] is False
    assert not path.exists()
    assert "private-secret" not in json.dumps(preview)
    with pytest.raises(ValueError, match="approval_token"):
        provenance_config.apply_config_update(update=update, approval_token="", path=path)

    applied = provenance_config.apply_config_update(
        update=update,
        approval_token=provenance_config.APPLY_APPROVAL_TOKEN,
        path=path,
    )

    assert applied["applied"] is True
    assert path.exists()
    assert "private-secret" not in json.dumps(applied)
    assert "private-secret" in path.read_text(encoding="utf-8")


def test_apply_update_preserves_contact_aliases(tmp_path: Path) -> None:
    path = tmp_path / "provenance.config.json"
    path.write_text(
        json.dumps(
            {
                "active_profile": "default",
                "contacts": {
                    "canonical_aliases": [
                        {
                            "id": "operator-eric",
                            "label": "Eric Cochran",
                            "emails": ["eric@saberchemical.com"],
                        }
                    ]
                },
                "profiles": {"default": {"source_ids": []}},
                "sources": {},
            }
        ),
        encoding="utf-8",
    )

    provenance_config.apply_config_update(
        update={"profiles": {"default": {"description": "updated"}}},
        approval_token=provenance_config.APPLY_APPROVAL_TOKEN,
        path=path,
    )

    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["contacts"]["canonical_aliases"][0]["label"] == "Eric Cochran"
